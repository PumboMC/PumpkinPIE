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

/// An answer built while a nested ask was refused still holds that
/// placeholder as written. Before, it was cached for the whole TTL.
#[test]
fn an_answer_built_around_a_refused_ask_is_not_cached() {
    let state = State::default();
    let on_request = |exp: &Exp<'_>| Some(text_of(exp.call(set_placeholders("<%exp_loop%>"))));
    let exp = Exp::new(&state, &on_request);
    register_as(&exp, Cache::Ttl { ms: 60_000 });

    let first = resolve(&exp, "%exp_loop%");
    let asks = exp.asks.get();
    assert_eq!(resolve(&exp, "%exp_loop%"), first);
    assert_eq!(exp.asks.get(), 2 * asks, "the second line asked again");
}

/// Two callers miss the same placeholder and the later one is answered
/// first. Before, the older answer arrived last and replaced it.
#[test]
fn an_older_answer_does_not_replace_a_newer_one() {
    let state = State::default();
    let asked = Cell::new(0);
    let on_request = |exp: &Exp<'_>| {
        asked.set(asked.get() + 1);
        if asked.get() > 1 {
            return Some("new".to_string());
        }
        let line = text_of(handle(
            exp.state,
            exp,
            "Scoreboard",
            set_placeholders("%exp_rank%"),
        ));
        assert_eq!(line, "new");
        Some("old".to_string())
    };
    let exp = Exp::new(&state, &on_request);
    register_as(&exp, Cache::Ttl { ms: 60_000 });

    assert_eq!(resolve(&exp, "%exp_rank%"), "old");
    assert_eq!(resolve(&exp, "%exp_rank%"), "new");
    assert_eq!(asked.get(), 2, "the newer answer came from the cache");
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

/// A Scoreboard line with seventeen uncached placeholders is refused at the
/// seventeenth while another caller waits on an answer. Before, any refusal
/// kept every answer in flight out of the cache, whoever it was for.
#[test]
fn a_refused_line_does_not_stop_other_callers_caching() {
    let state = State::default();
    let seventeen: String = (0..=MAX_ASKS_PER_LINE)
        .map(|n| format!("%exp_s{n}% "))
        .collect();
    let first = Cell::new(true);
    let on_request = |exp: &Exp<'_>| {
        if first.replace(false) {
            let line = text_of(handle(
                exp.state,
                exp,
                "Scoreboard",
                set_placeholders(&seventeen),
            ));
            assert!(line.contains("%exp_s16%"), "the seventeenth was refused");
        }
        Some("value".to_string())
    };
    let exp = Exp::new(&state, &on_request);
    register_as(&exp, Cache::Ttl { ms: 60_000 });

    assert_eq!(resolve(&exp, "%exp_rank%"), "value");
    let asks = exp.asks.get();
    assert_eq!(resolve(&exp, "%exp_rank%"), "value");
    assert_eq!(exp.asks.get(), asks, "the first answer was cached");
}
