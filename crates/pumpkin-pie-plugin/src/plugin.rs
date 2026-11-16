//! The PumpkinPIE provider plugin.
//!
//! This crate is the Wasm component you drop into a server's `plugins/`
//! directory. It owns the expansion registry, answers `set_placeholders`, caches
//! what expansions allow it to, and exposes `/pie` for checking a config by hand.
//!
//! The protocol, the token scanner and the client live in `pumpkin-pie`, which
//! is a plain library. They are split because every plugin exports a symbol
//! called `init-plugin`, so a plugin that consumes PumpkinPIE cannot also link
//! a crate that exports one.
//!
//! The `Plugin` trait hands every callback `&self`, and command handlers must be
//! `'static`, so the state lives in a [`OnceLock`] rather than on the plugin
//! value. It is only touched inside [`crate::state`], which never calls the
//! host, so no lock is held across a host call: the host may deliver an IPC
//! callback back into this plugin before the call returns.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use pumpkin_pie::builtins::{self, ResolveContext};
use pumpkin_pie::expansion::Expansion;
use pumpkin_pie::protocol::{
    MAX_BATCH_LINES, MAX_ID_LENGTH, MAX_MESSAGE_BYTES, MAX_TEXT_LENGTH, ProtocolError,
    RegisteredPlaceholder, Request, ResolvedLine, Response, Success, decode_request, encode,
};
use pumpkin_pie::{IpcMessage, PluginId};
use pumpkin_plugin_api::{
    Context, Player, Plugin, PluginMetadata, Result, Server, register_plugin,
};
use tracing::{info, warn};

use crate::state::{Busy, Chain, Lookup, MAX_ASKS_PER_LINE, MAX_NESTED_ASKS, Refusal, State};

/// Version reported by `ping`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// How long the same warning about the same expansion stays quiet.
const WARN_EVERY: Duration = Duration::from_secs(60);

/// What the IPC import returns: the outer error means the plugin was not
/// reachable, the inner one that it refused.
type Reply = Result<Result<IpcMessage, String>, ()>;

/// How the provider messages an expansion. The plugin goes through the host;
/// the tests answer from a fake expansion that can call straight back in.
pub(crate) trait Ipc {
    fn send(&self, recipient: &str, message: &[u8]) -> Reply;
}

/// The server's inter-plugin IPC.
struct HostIpc;

impl Ipc for HostIpc {
    fn send(&self, recipient: &str, message: &[u8]) -> Reply {
        pumpkin_plugin_api::ipc::send_ipc_message(recipient, message)
    }
}

static STATE: OnceLock<State> = OnceLock::new();
static SERVER: OnceLock<Server> = OnceLock::new();

/// When each warning was last logged, by what it is about. Its lock is held only
/// inside [`should_warn`], which calls nothing, so it is never contended.
static WARNED: Mutex<Option<HashMap<String, Instant>>> = Mutex::new(None);

/// The provider's state.
fn state() -> &'static State {
    STATE.get_or_init(State::default)
}

/// The server handle `on_load` was given.
///
/// # Panics
/// Panics if called before `on_load`.
pub fn server() -> &'static Server {
    SERVER
        .get()
        .expect("the server handle is only valid after on_load")
}

/// Whether a warning about `about` is due: the first one, then one a minute,
/// so a misconfiguration that repeats on every refresh is named without
/// flooding the log.
fn should_warn(about: &str, now: Instant) -> bool {
    let mut warned = WARNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let warned = warned.get_or_insert_with(HashMap::new);
    if warned
        .get(about)
        .is_some_and(|last| now.saturating_duration_since(*last) < WARN_EVERY)
    {
        return false;
    }
    warned.insert(about.to_string(), now);
    true
}

/// Refuses a call that found the state locked, rather than trapping on it.
fn busy(about: &str) -> Response {
    if should_warn(&format!("busy {about}"), Instant::now()) {
        warn!("provider state was locked by a call still in progress; refused {about}");
    }
    Response::failure("PumpkinPIE is busy, try again")
}

/// Resolves one line, returning the text and what stayed unresolved.
pub fn resolve_line(viewer_name: Option<&str>, text: &str) -> ResolvedLine {
    try_resolve_line(viewer_name, text).unwrap_or_else(|_| ResolvedLine {
        text: text.to_string(),
        unresolved: Vec::new(),
    })
}

pub(crate) fn try_resolve_line(
    viewer_name: Option<&str>,
    text: &str,
) -> Result<ResolvedLine, ProtocolError> {
    resolve_text(state(), &HostIpc, None, viewer_name, text)
}

/// Resolves one line for `sender`, the plugin that asked, or `None` for the
/// server itself.
fn resolve_text(
    state: &State,
    ipc: &dyn Ipc,
    sender: Option<&str>,
    viewer_name: Option<&str>,
    text: &str,
) -> Result<ResolvedLine, ProtocolError> {
    let viewer = viewer_name.and_then(|name| server().get_player_by_name(name));
    let viewer_key = viewer_name.unwrap_or_default().to_string();
    let chain = state.chain(sender);

    let result = pumpkin_pie::tokens::try_substitute(text, MAX_MESSAGE_BYTES, |id, argument| {
        resolve_one(
            state,
            ipc,
            &chain,
            viewer.as_ref(),
            &viewer_key,
            id,
            argument,
        )
    })
    .map_err(|_| ProtocolError::TooLarge)?;
    Ok(ResolvedLine {
        text: result.text,
        unresolved: result.unresolved,
    })
}

/// Resolves a single placeholder, or `None` to leave it alone.
fn resolve_one(
    state: &State,
    ipc: &dyn Ipc,
    chain: &Chain<'_>,
    viewer: Option<&Player>,
    viewer_key: &str,
    id: &str,
    argument: Option<&str>,
) -> Option<String> {
    if let Some(builtin) = builtins::find(id) {
        let context = ResolveContext {
            server: server(),
            viewer,
            argument,
        };
        return (builtin.resolve)(&context);
    }

    let now = Instant::now();
    let pending = match state.lookup(chain, id, viewer_key, argument, now) {
        Ok(Lookup::Unknown) => return None,
        Ok(Lookup::Cached(value)) => return value,
        Ok(Lookup::Ask(pending)) => pending,
        Ok(Lookup::Refused { source, why }) => {
            if should_warn(&format!("{why:?} {source}"), now) {
                match why {
                    Refusal::Budget => warn!(
                        "leaving {id} alone: one line may ask expansions {MAX_ASKS_PER_LINE} \
                         times, calls back included, and {source} was next"
                    ),
                    Refusal::Nesting => warn!(
                        "leaving {id} alone: {source} keeps resolving placeholders through \
                         PumpkinPIE while it is being asked, {MAX_NESTED_ASKS} deep"
                    ),
                }
            }
            return None;
        }
        Err(Busy) => {
            busy(id);
            return None;
        }
    };

    let request = callback_request(
        &pending.expansion,
        id,
        viewer.map(|player| player.get_name()),
        argument,
    )?;
    let value = ask_expansion(ipc, &pending.expansion.source, &request);
    // The state decides whether the answer may still be cached: the expansion
    // may have registered again meanwhile.
    if pending.store(value.clone()).is_err() {
        busy(id);
    }
    value
}

fn callback_request(
    expansion: &Expansion,
    id: &str,
    viewer: Option<String>,
    argument: Option<&str>,
) -> Option<Request> {
    let (_, name) = id.split_once('_')?;
    Some(Request::OnRequest {
        namespace: expansion.namespace.clone(),
        id: id.to_string(),
        // Lowercased to match how `Registry::add` stores names. Sending it as
        // written meant `%myexp_COUNT%` reached the expansion as `COUNT`, which
        // no `match` arm for the registered `count` would ever hit, so the
        // placeholder silently stayed in the text.
        name: name.to_ascii_lowercase(),
        viewer,
        argument: argument.map(str::to_string),
    })
}

/// Every placeholder the provider knows, built in and contributed.
pub fn registered() -> Vec<RegisteredPlaceholder> {
    registered_in(state()).unwrap_or_else(|Busy| {
        busy("the /pie command");
        Vec::new()
    })
}

fn registered_in(state: &State) -> Result<Vec<RegisteredPlaceholder>, Busy> {
    let mut entries: Vec<RegisteredPlaceholder> = builtins::BUILTINS
        .iter()
        .map(|builtin| RegisteredPlaceholder {
            id: builtin.id.to_string(),
            description: builtin.description.to_string(),
            namespace: namespace_of(builtin.id).to_string(),
            source: pumpkin_pie::PROVIDER.to_string(),
        })
        .collect();

    for expansion in state.expansions()? {
        for name in &expansion.names {
            entries.push(RegisteredPlaceholder {
                id: format!("{}_{}", expansion.namespace, name),
                description: String::new(),
                namespace: expansion.namespace.clone(),
                source: expansion.source.clone(),
            });
        }
    }
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    entries.dedup_by(|left, right| left.id == right.id);
    Ok(entries)
}

/// One line per registered expansion, for `/pie`.
pub fn expansion_summary() -> Vec<String> {
    let Ok(expansions) = state().expansions() else {
        busy("the /pie command");
        return Vec::new();
    };
    expansions
        .iter()
        .map(|expansion| {
            let names = if expansion.names.is_empty() {
                "answers anything".to_string()
            } else {
                expansion
                    .names
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!(
                "%{}%  from {}: {names}",
                expansion.namespace, expansion.source
            )
        })
        .collect()
}

/// Handles one decoded request.
fn handle(state: &State, ipc: &dyn Ipc, sender: &str, request: Request) -> Response {
    match request {
        Request::Ping => {
            let Ok(expansions) = state.expansions() else {
                return busy(sender);
            };
            Response::success(Success::Ping {
                protocol: pumpkin_pie::PROTOCOL_VERSION,
                name: pumpkin_pie::PROVIDER.to_string(),
                version: VERSION.to_string(),
                placeholders: builtins::count(),
                expansions: expansions.len(),
            })
        }
        Request::GetRegisteredPlaceholders => match registered_in(state) {
            Ok(placeholders) => Response::success(Success::Registered { placeholders }),
            Err(Busy) => busy(sender),
        },
        Request::RegisterExpansion {
            namespace,
            placeholders,
            cache,
        } => match state.register(sender, &namespace, placeholders, cache) {
            Ok(Ok(expansion)) => Response::success(Success::RegisteredExpansion {
                namespace: expansion.namespace,
                placeholders: expansion.names.into_iter().collect(),
                cache: expansion.cache,
            }),
            Ok(Err(error)) => Response::from_error(error),
            Err(Busy) => busy(sender),
        },
        Request::UnregisterExpansion => {
            let Ok(namespaces) = state.unregister(sender) else {
                return busy(sender);
            };
            for namespace in &namespaces {
                info!("{sender} released {namespace}");
            }
            Response::success(Success::Unregistered { namespaces })
        }
        Request::SetPlaceholders { text, viewer, .. } => {
            if let Err(error) = check_text(&text) {
                return Response::from_error(error);
            }
            match resolve_text(state, ipc, Some(sender), viewer.as_deref(), &text) {
                Ok(line) => Response::success(Success::Resolved(line)),
                Err(error) => Response::from_error(error),
            }
        }
        Request::SetPlaceholdersBatch { requests } => {
            if requests.len() > MAX_BATCH_LINES {
                return Response::failure(format!("at most {MAX_BATCH_LINES} lines per batch"));
            }
            resolve_batch(
                &requests,
                |line| resolve_text(state, ipc, Some(sender), line.viewer.as_deref(), &line.text),
                MAX_MESSAGE_BYTES,
            )
        }
        Request::GetPlaceholderValue {
            id,
            viewer,
            argument,
        } => {
            if id.len() > MAX_ID_LENGTH {
                return Response::from_error(ProtocolError::TooLong {
                    field: "id".to_string(),
                    limit: MAX_ID_LENGTH,
                });
            }
            let chain = state.chain(Some(sender));
            let viewer_key = viewer.clone().unwrap_or_default();
            let resolved = viewer
                .as_deref()
                .and_then(|name| server().get_player_by_name(name));
            match resolve_one(
                state,
                ipc,
                &chain,
                resolved.as_ref(),
                &viewer_key,
                &id,
                argument.as_deref(),
            ) {
                Some(value) => Response::success(Success::Value {
                    known: true,
                    value: Some(value),
                }),
                None => Response::success(Success::Value {
                    known: false,
                    value: None,
                }),
            }
        }
        Request::OnRequest { .. } => Response::failure("the provider does not provide expansions"),
    }
}

/// The reserved namespace a built in belongs to.
fn namespace_of(id: &str) -> &str {
    id.split_once('_').map_or(id, |(namespace, _)| namespace)
}

fn check_text(text: &str) -> Result<(), ProtocolError> {
    if text.len() > MAX_TEXT_LENGTH {
        return Err(ProtocolError::TooLong {
            field: "text".to_string(),
            limit: MAX_TEXT_LENGTH,
        });
    }
    Ok(())
}

fn resolve_batch(
    requests: &[pumpkin_pie::protocol::Line],
    mut resolve: impl FnMut(&pumpkin_pie::protocol::Line) -> Result<ResolvedLine, ProtocolError>,
    limit: usize,
) -> Response {
    for line in requests {
        if let Err(error) = check_text(&line.text) {
            return Response::from_error(error);
        }
    }
    // Reserve space for the response wrapper and separators. Each line is
    // serialized separately to account for JSON escaping as well as text.
    let mut remaining = limit.saturating_sub(64);
    let mut results = Vec::new();
    for line in requests {
        let result = match resolve(line) {
            Ok(result) => result,
            Err(error) => return Response::from_error(error),
        };
        let size = match encode(&result) {
            Ok(bytes) => bytes.len().saturating_add(1),
            Err(error) => return Response::from_error(error),
        };
        if size > remaining {
            return Response::from_error(ProtocolError::TooLarge);
        }
        remaining -= size;
        results.push(result);
    }
    Response::success(Success::ResolvedBatch { results })
}

fn encode_response(response: &Response) -> Vec<u8> {
    encode(response).unwrap_or_else(|_| {
        // This fixed error is always below the protocol limit.
        br#"{"ok":false,"error":"response is too large"}"#.to_vec()
    })
}

/// Asks the plugin that owns an expansion for one of its placeholders.
fn ask_expansion(ipc: &dyn Ipc, source: &str, request: &Request) -> Option<String> {
    let bytes = encode(request).ok()?;
    let reply = match ipc.send(source, &bytes) {
        Ok(Ok(reply)) => reply,
        Ok(Err(message)) => {
            warn!("{source} refused an on_request: {message}");
            return None;
        }
        Err(()) => {
            warn!("{source} did not answer an on_request");
            return None;
        }
    };
    match pumpkin_pie::protocol::decode_response(&reply)
        .ok()?
        .into_success()
    {
        Ok(Success::OnRequest { value }) => value,
        Ok(_) => {
            warn!("{source} answered an on_request with the wrong response");
            None
        }
        Err(message) => {
            warn!("{source} failed an on_request: {message}");
            None
        }
    }
}

pub struct PumpkinPie;

impl Plugin for PumpkinPie {
    fn new() -> Self {
        Self
    }

    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: pumpkin_pie::PROVIDER.to_string(),
            version: VERSION.to_string(),
            authors: vec!["Rennex".to_string()],
            description: "Placeholder provider that other Pumpkin plugins resolve through"
                .to_string(),
            dependencies: Vec::new(),
            permissions: Vec::new(),
        }
    }

    fn on_load(&self, context: Context) -> Result<()> {
        let _ = SERVER.set(context.get_server());
        crate::command::register(context);
        info!(
            "PumpkinPIE {VERSION} ready with {} built in placeholders",
            builtins::count()
        );
        Ok(())
    }

    fn on_unload(&self, _context: Context) -> Result<()> {
        info!("PumpkinPIE unloaded");
        Ok(())
    }

    fn handle_ipc_message(
        &self,
        sender: PluginId,
        message: IpcMessage,
    ) -> std::result::Result<IpcMessage, String> {
        let request = match decode_request(&message) {
            Ok(request) => request,
            Err(error) => {
                return Ok(encode(&Response::from_error(error)).unwrap_or_default());
            }
        };
        let response = handle(state(), &HostIpc, &sender, request);
        Ok(encode_response(&response))
    }
}

register_plugin!(PumpkinPie);

#[cfg(test)]
mod tests {
    use super::*;
    use pumpkin_pie::answer;
    use pumpkin_pie::expansion::Registry;
    use pumpkin_pie::protocol::{Cache, Line, decode_response};
    use std::cell::Cell;

    /// The plugin `Exp` on a fake server. The provider asks it over [`Ipc`], it
    /// answers through the same `answer` an expansion embeds, and `call` reaches
    /// the provider's handler while the ask is still open, which is what
    /// Pumpkin does when an asked plugin messages the provider back.
    struct Exp<'a> {
        state: &'a State,
        on_request: &'a dyn Fn(&Exp<'a>) -> Option<String>,
        asks: Cell<usize>,
        depth: Cell<usize>,
    }

    impl Ipc for Exp<'_> {
        fn send(&self, recipient: &str, message: &[u8]) -> Reply {
            assert_eq!(recipient, "Exp");
            self.asks.set(self.asks.get() + 1);
            let reply = self.nest(|| answer(message, |_| (self.on_request)(self)));
            Ok(Ok(reply.expect("the reply encodes")))
        }
    }

    impl<'a> Exp<'a> {
        fn new(state: &'a State, on_request: &'a dyn Fn(&Exp<'a>) -> Option<String>) -> Self {
            Self {
                state,
                on_request,
                asks: Cell::new(0),
                depth: Cell::new(0),
            }
        }

        /// One more hop on the call chain. Pumpkin allows 64 and traps the
        /// plugin that makes the next one.
        fn nest<T>(&self, call: impl FnOnce() -> T) -> T {
            assert!(self.depth.get() < 64, "Pumpkin traps a plugin here");
            self.depth.set(self.depth.get() + 1);
            let result = call();
            self.depth.set(self.depth.get() - 1);
            result
        }

        /// `Exp` messaging the provider.
        fn call(&self, request: Request) -> Response {
            self.nest(|| handle(self.state, self, "Exp", request))
        }

        fn register(&self) {
            let response = self.call(Request::RegisterExpansion {
                namespace: "exp".to_string(),
                placeholders: Vec::new(),
                cache: Cache::Ttl { ms: 60_000 },
            });
            assert!(response.ok, "{response:?}");
        }
    }

    fn set_placeholders(text: &str) -> Request {
        Request::SetPlaceholders {
            text: text.to_string(),
            viewer: None,
            argument: None,
        }
    }

    fn text_of(response: Response) -> String {
        match response.into_success() {
            Ok(Success::Resolved(line)) => line.text,
            other => panic!("expected a resolved line, got {other:?}"),
        }
    }

    /// Another plugin, say a tab list, resolving `text`.
    fn resolve(exp: &Exp<'_>, text: &str) -> String {
        text_of(handle(exp.state, exp, "TabList", set_placeholders(text)))
    }

    /// Re-entry as the v0.1 API does it: `Exp` answers by resolving its own
    /// placeholder through the provider, which is still waiting on that answer.
    /// Nothing stopped this before, so it recursed until Pumpkin trapped a
    /// plugin at 64 nested calls.
    #[test]
    fn an_expansion_resolving_its_own_placeholder_stops_nesting() {
        let state = State::default();
        let on_request = |exp: &Exp<'_>| Some(text_of(exp.call(set_placeholders("<%exp_loop%>"))));
        let exp = Exp::new(&state, &on_request);
        exp.register();

        assert_eq!(resolve(&exp, "%exp_loop%"), "<<<<<%exp_loop%>>>>>");
        assert_eq!(exp.asks.get(), 1 + MAX_NESTED_ASKS);
    }

    /// `Exp` reloads and registers again from inside its own answer, which is
    /// how the v0.1 API runs a call while an ask is open; a reload under the
    /// concurrent v0.2 API lands at the same point. The answer computed before
    /// the reload arrives after it. Before, that answer was cached under the
    /// new registration and served until its TTL ran out.
    #[test]
    fn an_answer_from_before_a_reregistration_is_not_cached() {
        let state = State::default();
        let value = Cell::new("old");
        let on_request = |exp: &Exp<'_>| {
            let answer = value.replace("new");
            if answer == "old" {
                exp.register();
            }
            Some(answer.to_string())
        };
        let exp = Exp::new(&state, &on_request);
        exp.register();

        assert_eq!(resolve(&exp, "%exp_rank%"), "old");
        assert_eq!(resolve(&exp, "%exp_rank%"), "new");
        assert_eq!(resolve(&exp, "%exp_rank%"), "new");
        assert_eq!(exp.asks.get(), 2, "the answer after the reload is cached");
    }

    /// If a change ever holds the state across a call back in, that call is
    /// refused. Locking twice instead would panic on Wasm and trap the plugin.
    #[test]
    fn a_locked_state_refuses_instead_of_trapping() {
        let state = State::default();
        let on_request = |_: &Exp<'_>| Some("value".to_string());
        let exp = Exp::new(&state, &on_request);
        exp.register();
        let _held = state.hold();

        assert_eq!(resolve(&exp, "%exp_rank%"), "%exp_rank%");
        let response = handle(&state, &exp, "Exp", Request::UnregisterExpansion);
        assert_eq!(
            response.error.as_deref(),
            Some("PumpkinPIE is busy, try again")
        );
        assert_eq!(exp.asks.get(), 0);
    }

    fn register_as(exp: &Exp<'_>, cache: Cache) {
        let response = exp.call(Request::RegisterExpansion {
            namespace: "exp".to_string(),
            placeholders: Vec::new(),
            cache,
        });
        assert!(response.ok, "{response:?}");
    }

    /// Under the concurrent v0.2 API callers wait on expansions side by side,
    /// played here by each new caller arriving while the previous one waits.
    /// Before, four of them waiting sent the next caller's line back as
    /// written, because every ask on the server counted towards the limit.
    #[test]
    fn callers_waiting_side_by_side_do_not_stop_another_caller() {
        let state = State::default();
        let callers = Cell::new(1);
        let on_request = |exp: &Exp<'_>| {
            let caller = callers.get();
            if caller <= MAX_NESTED_ASKS {
                callers.set(caller + 1);
                let caller = format!("TabList{caller}");
                let line = text_of(handle(
                    exp.state,
                    exp,
                    &caller,
                    set_placeholders("%exp_rank%"),
                ));
                assert_eq!(line, "value", "{caller} was refused");
            }
            Some("value".to_string())
        };
        let exp = Exp::new(&state, &on_request);
        register_as(&exp, Cache::Never);

        assert_eq!(resolve(&exp, "%exp_rank%"), "value");
        assert_eq!(exp.asks.get(), MAX_NESTED_ASKS + 1);
    }

    /// An expansion answering each placeholder by resolving sixteen more of its
    /// own got a fresh budget at every level before, so one line cost 69,904
    /// asks. Calls nested under a line now share its budget.
    #[test]
    fn calls_nested_under_a_line_share_its_budget() {
        let state = State::default();
        let sixteen: String = (0..16).map(|n| format!("%exp_n{n}% ")).collect();
        let on_request = |exp: &Exp<'_>| {
            text_of(exp.call(set_placeholders(&sixteen)));
            Some("x".to_string())
        };
        let exp = Exp::new(&state, &on_request);
        register_as(&exp, Cache::Never);

        resolve(&exp, &sixteen);
        assert!(
            exp.asks.get() <= MAX_ASKS_PER_LINE,
            "{} asks",
            exp.asks.get()
        );
    }

    /// Before, each warning was logged once for the plugin's whole life and
    /// never named the expansion, so a second one misbehaving stayed silent.
    #[test]
    fn warnings_repeat_once_a_minute_for_each_expansion() {
        let now = Instant::now();
        assert!(should_warn("test Exp", now));
        assert!(!should_warn("test Exp", now + Duration::from_secs(59)));
        assert!(should_warn("test Other", now + Duration::from_secs(59)));
        assert!(should_warn("test Exp", now + WARN_EVERY));
    }

    #[test]
    fn callback_uses_registered_normalized_namespace() {
        let mut registry = Registry::new();
        let expansion = registry.add("Ranks", "source", [], Cache::Never).unwrap();
        let request = callback_request(expansion, "RANKS_Prefix", None, None).unwrap();
        assert!(
            matches!(request, Request::OnRequest { namespace, name, id, .. } if namespace == "ranks" && name == "prefix" && id == "RANKS_Prefix")
        );
    }

    #[test]
    fn batch_rejects_oversized_middle_line_before_resolving() {
        let lines = vec![
            Line {
                viewer: Some("a".into()),
                text: "first".into(),
            },
            Line {
                viewer: Some("b".into()),
                text: "x".repeat(MAX_TEXT_LENGTH + 1),
            },
            Line {
                viewer: Some("c".into()),
                text: "last".into(),
            },
        ];
        let mut calls = 0;
        let response = resolve_batch(
            &lines,
            |_| {
                calls += 1;
                unreachable!()
            },
            MAX_MESSAGE_BYTES,
        );
        assert!(!response.ok);
        assert_eq!(calls, 0);
        assert!(!decode_response(&encode_response(&response)).unwrap().ok);
    }

    #[test]
    fn valid_batch_preserves_viewer_order_and_cardinality() {
        let lines: Vec<Line> = ["a", "b", "c"]
            .into_iter()
            .map(|name| Line {
                viewer: Some(name.into()),
                text: "x".into(),
            })
            .collect();
        let response = resolve_batch(
            &lines,
            |line| {
                Ok(ResolvedLine {
                    text: line.viewer.clone().unwrap(),
                    unresolved: vec![],
                })
            },
            MAX_MESSAGE_BYTES,
        );
        let Success::ResolvedBatch { results } = response.into_success().unwrap() else {
            panic!("expected batch");
        };
        assert_eq!(
            results
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn batch_limits_aggregate_encoded_size_and_returns_valid_failure() {
        let lines = vec![
            Line {
                viewer: None,
                text: "x".into()
            };
            3
        ];
        let response = resolve_batch(
            &lines,
            |_| {
                Ok(ResolvedLine {
                    text: "\"".repeat(20),
                    unresolved: vec![],
                })
            },
            150,
        );
        assert!(!response.ok);
        assert!(!decode_response(&encode_response(&response)).unwrap().ok);
        let oversized = Response::success(Success::Value {
            known: true,
            value: Some("x".repeat(MAX_MESSAGE_BYTES)),
        });
        assert!(!decode_response(&encode_response(&oversized)).unwrap().ok);
    }
}
