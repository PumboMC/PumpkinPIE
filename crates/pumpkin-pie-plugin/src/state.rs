//! What the provider remembers, and the only code that touches it.
//!
//! The lock is taken in [`State::with`] and nowhere else, and nothing in this
//! module calls the server: it imports no host API, and the clock is passed in.
//! So no guard can be held across a host call. That matters because a host call
//! can bring a call back into this plugin before it returns: an expansion being
//! asked over IPC may call the provider, and the concurrent v0.2 API will run
//! calls side by side. On Wasm, locking a held mutex again panics, which traps
//! the plugin, so [`State::with`] answers [`Busy`] instead.

use std::collections::HashMap;
use std::sync::{Mutex, TryLockError};
use std::time::{Duration, Instant};

use pumpkin_pie::expansion::{Expansion, Registry};
use pumpkin_pie::protocol::{Cache, MAX_MESSAGE_BYTES, ProtocolError};

/// Expansion asks one top-level line may make, nested calls included.
///
/// An expansion is asked synchronously, so a misbehaving one could otherwise
/// ask for placeholders forever. This is the budget for one resolved line, not
/// one message, so a batch of a thousand lines is not penalised for being
/// large. Calls nested under the line share it, so an expansion that answers by
/// resolving more of its own placeholders cannot multiply it.
pub(crate) const MAX_ASKS_PER_LINE: usize = 16;

/// Asks that the calls nested under one line may have waiting at once.
///
/// A call is nested when it comes from an expansion the provider is waiting
/// on. Pumpkin 0.2.0 to 0.2.2 deliver it within the same call chain and trap the
/// plugin that makes the 65th nested call, so without a limit an expansion
/// whose answer resolves its own placeholder recursed until a plugin trapped.
/// Calls from anyone else are never held to it, so callers waiting side by side
/// under the concurrent v0.2 API do not refuse each other. A loop through a
/// third plugin does not look like nesting from here; [`MAX_WAITING_ASKS`]
/// stops that one.
pub(crate) const MAX_NESTED_ASKS: usize = 4;

/// Asks that may be waiting on expansions at once, across every chain.
///
/// A backstop for loops [`MAX_NESTED_ASKS`] cannot see, such as an expansion
/// that messages another plugin which resolves the expansion's placeholder
/// through the provider: every such call starts a fresh chain. Each turn of
/// that loop takes three of the 64 nested calls Pumpkin allows, so sixteen
/// asks stop it well inside the limit, and the concurrent v0.2 API rarely has
/// this many waiting.
pub(crate) const MAX_WAITING_ASKS: usize = 16;

/// What one expansion's value is cached against.
type CacheKey = (String, String, String, Option<String>);

/// One expansion's cached values, keyed by viewer and placeholder.
type ExpansionCache = HashMap<CacheKey, Cached>;

const MAX_CACHE_ENTRIES: usize = 4096;
const MAX_CACHE_BYTES: usize = 4 * MAX_MESSAGE_BYTES;

/// A cached expansion value, and when it was stored.
struct Cached {
    value: Option<String>,
    stored: Instant,
    ttl: Duration,
}

fn entry_size(key: &CacheKey, value: &Option<String>) -> usize {
    key.0.len()
        + key.1.len()
        + key.2.len()
        + key.3.as_ref().map_or(0, String::len)
        + value.as_ref().map_or(0, String::len)
}

fn cache_key(source: &str, viewer: &str, id: &str, argument: Option<&str>) -> CacheKey {
    (
        source.to_string(),
        viewer.to_string(),
        id.to_ascii_lowercase(),
        argument.map(str::to_string),
    )
}

fn cache_get(cache: &mut ExpansionCache, key: &CacheKey, now: Instant) -> Option<Option<String>> {
    let entry = cache.get(key)?;
    if now.saturating_duration_since(entry.stored) >= entry.ttl {
        cache.remove(key);
        return None;
    }
    Some(entry.value.clone())
}

fn cache_insert(
    cache: &mut ExpansionCache,
    key: CacheKey,
    value: Option<String>,
    ttl: Duration,
    now: Instant,
    max_entries: usize,
    max_bytes: usize,
) {
    cache.retain(|_, item| now.saturating_duration_since(item.stored) < item.ttl);
    cache.remove(&key);
    let size = entry_size(&key, &value);
    if max_entries == 0 || size > max_bytes {
        return;
    }
    let mut used: usize = cache
        .iter()
        .map(|(key, item)| entry_size(key, &item.value))
        .sum();
    while cache.len() >= max_entries || size > max_bytes.saturating_sub(used) {
        let Some(oldest) = cache
            .iter()
            .min_by_key(|(_, item)| item.stored)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        if let Some(item) = cache.remove(&oldest) {
            used -= entry_size(&oldest, &item.value);
        }
    }
    cache.insert(
        key,
        Cached {
            value,
            stored: now,
            ttl,
        },
    );
}

/// Everything the provider owns.
#[derive(Default)]
pub(crate) struct State {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    registry: Registry,
    /// Keyed by expansion source, then by viewer and placeholder.
    cache: ExpansionCache,
    /// Moves on every register and unregister, and when [`MAX_WAITING_ASKS`]
    /// refuses an ask, so an answer in flight across one is not cached: it may
    /// belong to an old registration, or hold a placeholder left as written by
    /// a loop that runs through several chains. One counter for everything: a
    /// change anywhere only costs the answers in flight their cache entry.
    generation: u64,
    /// What each call chain has left, by chain id.
    chains: HashMap<u64, Budget>,
    last_chain: u64,
    /// For each expansion being asked, the chain of every ask waiting on it,
    /// newest last.
    waiting: HashMap<String, Vec<u64>>,
}

/// What a top-level line and the calls nested under it have left.
struct Budget {
    asks_left: usize,
    nested_waiting: usize,
    /// Moves when a nested call in this chain is refused, so the answers in
    /// flight in it, which may hold that placeholder as written, are not cached.
    refused: u64,
}

/// The state was already locked, which only a call coming back in while the
/// lock was held can cause.
#[derive(Debug)]
pub(crate) struct Busy;

/// Why an expansion was not asked.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Refusal {
    /// The line spent [`MAX_ASKS_PER_LINE`].
    Budget,
    /// [`MAX_NESTED_ASKS`] nested asks were already waiting.
    Nesting,
    /// [`MAX_WAITING_ASKS`] asks were already waiting, across every chain.
    Backstop,
}

/// What resolving an expansion placeholder needs next.
pub(crate) enum Lookup<'a> {
    /// No expansion owns the namespace.
    Unknown,
    /// A value the expansion let the provider reuse.
    Cached(Option<String>),
    /// Leave the placeholder as written.
    Refused { source: String, why: Refusal },
    /// Ask the expansion, then hand the answer to [`Pending::store`].
    Ask(Pending<'a>),
}

/// The call chain a line resolves under: a new one for a top-level line, or the
/// chain of the ask waiting on the caller when an expansion calls back.
pub(crate) struct Chain<'a> {
    state: &'a State,
    id: u64,
    nested: bool,
}

impl Drop for Chain<'_> {
    fn drop(&mut self) {
        if !self.nested {
            let _ = self.state.with(|inner| inner.chains.remove(&self.id));
        }
    }
}

/// An ask about to be made, and what its answer may be cached against. It
/// counts as waiting on its expansion until it is dropped.
pub(crate) struct Pending<'a> {
    state: &'a State,
    pub(crate) expansion: Expansion,
    key: CacheKey,
    generation: u64,
    asked: Instant,
    chain: u64,
    refused: u64,
    nested: bool,
}

impl Pending<'_> {
    /// Caches the answer if the expansion asked for a TTL and nothing moved
    /// while it was asked: no registration, no refused ask in its chain, and no
    /// answer to a later ask for the same placeholder.
    pub(crate) fn store(self, value: Option<String>) -> Result<(), Busy> {
        let ttl = self.expansion.ttl_ms();
        self.state.with(|inner| {
            let newer = inner
                .cache
                .get(&self.key)
                .is_some_and(|entry| entry.stored > self.asked);
            let refused = inner
                .chains
                .get(&self.chain)
                .is_some_and(|budget| budget.refused != self.refused);
            if ttl == 0 || inner.generation != self.generation || newer || refused {
                return;
            }
            cache_insert(
                &mut inner.cache,
                self.key.clone(),
                value,
                Duration::from_millis(ttl),
                self.asked,
                MAX_CACHE_ENTRIES,
                MAX_CACHE_BYTES,
            );
        })
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        let _ = self.state.with(|inner| {
            if let Some(chains) = inner.waiting.get_mut(&self.expansion.source) {
                if let Some(index) = chains.iter().rposition(|chain| *chain == self.chain) {
                    chains.remove(index);
                }
                if chains.is_empty() {
                    inner.waiting.remove(&self.expansion.source);
                }
            }
            if self.nested {
                if let Some(budget) = inner.chains.get_mut(&self.chain) {
                    budget.nested_waiting -= 1;
                }
            }
        });
    }
}

impl State {
    /// The only place the lock is taken. `f` gets plain data and nothing that
    /// reaches the server.
    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> Result<R, Busy> {
        let mut inner = match self.inner.try_lock() {
            Ok(inner) => inner,
            // A poisoned lock is a bug, not a reason to stop resolving placeholders.
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return Err(Busy),
        };
        Ok(f(&mut inner))
    }

    /// Claims or updates `namespace` for `sender`, dropping everything cached for
    /// `sender`, which a re-register invalidates.
    pub(crate) fn register(
        &self,
        sender: &str,
        namespace: &str,
        names: Vec<String>,
        cache: Cache,
    ) -> Result<Result<Expansion, ProtocolError>, Busy> {
        self.with(|inner| {
            inner.cache.retain(|key, _| key.0 != sender);
            inner.generation += 1;
            inner.registry.add(namespace, sender, names, cache).cloned()
        })
    }

    /// Forgets every namespace `sender` claimed and what was cached for it.
    pub(crate) fn unregister(&self, sender: &str) -> Result<Vec<String>, Busy> {
        self.with(|inner| {
            inner.cache.retain(|key, _| key.0 != sender);
            inner.generation += 1;
            inner.registry.drop_source(sender)
        })
    }

    /// Every registered expansion, sorted by namespace.
    pub(crate) fn expansions(&self) -> Result<Vec<Expansion>, Busy> {
        self.with(|inner| inner.registry.all().cloned().collect())
    }

    /// The chain a line from `sender` resolves under; `None` is the server
    /// itself, such as `/pie`. A call from an expansion the provider is waiting
    /// on joins the chain of the newest ask waiting on it.
    pub(crate) fn chain(&self, sender: Option<&str>) -> Chain<'_> {
        let joined = self.with(|inner| {
            if let Some(&id) = sender.and_then(|sender| inner.waiting.get(sender)?.last()) {
                return (id, true);
            }
            inner.last_chain += 1;
            let budget = Budget {
                asks_left: MAX_ASKS_PER_LINE,
                nested_waiting: 0,
                refused: 0,
            };
            inner.chains.insert(inner.last_chain, budget);
            (inner.last_chain, false)
        });
        // A busy state refuses every lookup in this chain anyway.
        let (id, nested) = joined.unwrap_or((0, true));
        Chain {
            state: self,
            id,
            nested,
        }
    }

    /// Finds who answers `id` for `viewer`, and whether a cached value will do.
    /// An ask counts against `chain` until the [`Pending`] it returns is dropped.
    pub(crate) fn lookup(
        &self,
        chain: &Chain<'_>,
        id: &str,
        viewer: &str,
        argument: Option<&str>,
        now: Instant,
    ) -> Result<Lookup<'_>, Busy> {
        self.with(|inner| {
            let Some(expansion) = inner.registry.owner_of(id).cloned() else {
                return Lookup::Unknown;
            };
            let key = cache_key(&expansion.source, viewer, id, argument);
            if expansion.ttl_ms() > 0 {
                if let Some(value) = cache_get(&mut inner.cache, &key, now) {
                    return Lookup::Cached(value);
                }
            }
            let waiting: usize = inner.waiting.values().map(Vec::len).sum();
            let why = match inner.chains.get_mut(&chain.id) {
                _ if waiting >= MAX_WAITING_ASKS => Some(Refusal::Backstop),
                Some(budget) if budget.asks_left == 0 => Some(Refusal::Budget),
                Some(budget) if chain.nested && budget.nested_waiting >= MAX_NESTED_ASKS => {
                    Some(Refusal::Nesting)
                }
                Some(budget) => {
                    budget.asks_left -= 1;
                    budget.nested_waiting += usize::from(chain.nested);
                    None
                }
                None => Some(Refusal::Budget),
            };
            if let Some(why) = why {
                // Answers in flight may now hold this placeholder as written. A
                // top-level refusal has none in its chain, and the loop the
                // backstop stops runs through several chains.
                if matches!(why, Refusal::Backstop) {
                    inner.generation += 1;
                } else if chain.nested {
                    if let Some(budget) = inner.chains.get_mut(&chain.id) {
                        budget.refused += 1;
                    }
                }
                return Lookup::Refused {
                    source: expansion.source,
                    why,
                };
            }
            let refused = inner
                .chains
                .get(&chain.id)
                .map_or(0, |budget| budget.refused);
            inner
                .waiting
                .entry(expansion.source.clone())
                .or_default()
                .push(chain.id);
            Lookup::Ask(Pending {
                state: self,
                expansion,
                key,
                generation: inner.generation,
                asked: now,
                chain: chain.id,
                refused,
                nested: chain.nested,
            })
        })
    }

    /// Holds the lock, to stand in for a future change that keeps it across a
    /// call back into the plugin.
    #[cfg(test)]
    pub(crate) fn hold(&self) -> impl Sized + '_ {
        self.inner.lock().expect("the test state is not poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_distinguishes_arguments_and_negative_values() {
        let mut cache = ExpansionCache::new();
        let now = Instant::now();
        let ttl = Duration::from_secs(10);
        for (arg, value) in [
            (None, Some("none")),
            (Some(""), None),
            (Some("gold"), Some("10")),
            (Some("silver"), Some("50")),
        ] {
            cache_insert(
                &mut cache,
                cache_key("exp", "Steve", "RANKS_value", arg),
                value.map(str::to_string),
                ttl,
                now,
                10,
                1000,
            );
        }
        for (arg, value) in [
            (None, Some("none")),
            (Some(""), None),
            (Some("gold"), Some("10")),
            (Some("silver"), Some("50")),
        ] {
            assert_eq!(
                cache_get(
                    &mut cache,
                    &cache_key("exp", "Steve", "ranks_value", arg),
                    now
                ),
                Some(value.map(str::to_string))
            );
        }
    }

    #[test]
    fn cache_evicts_expired_and_oldest_entries_and_limits_bytes() {
        let mut cache = ExpansionCache::new();
        let now = Instant::now();
        let ttl = Duration::from_secs(2);
        let a = cache_key("s", "", "a", None);
        let b = cache_key("s", "", "b", None);
        let c = cache_key("s", "", "c", None);
        cache_insert(&mut cache, a.clone(), Some("x".into()), ttl, now, 2, 100);
        cache_insert(
            &mut cache,
            b.clone(),
            None,
            ttl,
            now + Duration::from_millis(1),
            2,
            100,
        );
        cache_insert(
            &mut cache,
            c.clone(),
            None,
            ttl,
            now + Duration::from_millis(2),
            2,
            100,
        );
        assert!(!cache.contains_key(&a));
        assert!(cache.contains_key(&b));
        cache_insert(
            &mut cache,
            a.clone(),
            None,
            ttl,
            now + ttl + Duration::from_millis(2),
            2,
            100,
        );
        assert!(!cache.contains_key(&c));
        assert_eq!(
            cache_get(&mut cache, &b, now + ttl + Duration::from_millis(1)),
            None
        );
        cache_insert(
            &mut cache,
            b.clone(),
            Some("too large".into()),
            ttl,
            now + ttl,
            2,
            3,
        );
        assert!(!cache.contains_key(&b));
    }

    #[test]
    fn cache_evicts_old_entries_to_fit_byte_budget() {
        let mut cache = ExpansionCache::new();
        let now = Instant::now();
        let ttl = Duration::from_secs(5);
        let a = cache_key("s", "", "a", None);
        let b = cache_key("s", "", "b", None);
        cache_insert(&mut cache, a.clone(), Some("123".into()), ttl, now, 10, 7);
        cache_insert(
            &mut cache,
            b.clone(),
            Some("456".into()),
            ttl,
            now + Duration::from_millis(1),
            10,
            7,
        );
        assert!(!cache.contains_key(&a));
        assert!(cache.contains_key(&b));
    }

    #[test]
    fn cache_removes_an_expired_entry_on_lookup() {
        let mut cache = ExpansionCache::new();
        let now = Instant::now();
        let ttl = Duration::from_secs(1);
        let key = cache_key("s", "viewer", "s_value", None);
        cache_insert(&mut cache, key.clone(), None, ttl, now, 2, 100);
        assert_eq!(cache_get(&mut cache, &key, now), Some(None));
        assert_eq!(cache_get(&mut cache, &key, now + ttl), None);
        assert!(!cache.contains_key(&key));
    }

    /// The interleaving the v0.2 API allows, one step at a time: a line looks
    /// the expansion up, the expansion registers again or leaves while the
    /// provider waits for it, and then the answer arrives. Before, the answer
    /// was cached anyway, under a registration it was not given for.
    #[test]
    fn an_answer_is_not_cached_when_its_expansion_changed_while_asked() {
        let now = Instant::now();
        let register = |state: &State| {
            let ttl = Cache::Ttl { ms: 60_000 };
            state
                .register("Exp", "exp", Vec::new(), ttl)
                .unwrap()
                .unwrap();
        };
        let unregister = |state: &State| {
            state.unregister("Exp").unwrap();
        };
        for change in [register as fn(&State), unregister] {
            let state = State::default();
            register(&state);
            let chain = state.chain(None);
            let Ok(Lookup::Ask(pending)) = state.lookup(&chain, "exp_rank", "", None, now) else {
                panic!("an uncached placeholder is asked for");
            };
            change(&state);
            pending.store(Some("old".into())).unwrap();
            assert!(state.inner.lock().unwrap().cache.is_empty());
        }
    }

    /// Only calls from an expansion the provider is waiting on count as nested,
    /// and a nested ask stops counting once it is answered.
    #[test]
    fn nested_asks_are_capped_per_chain_and_released() {
        let now = Instant::now();
        let state = State::default();
        let never = Cache::Never;
        state
            .register("Exp", "exp", Vec::new(), never)
            .unwrap()
            .unwrap();
        let ask = |chain: &Chain<'_>| state.lookup(chain, "exp_rank", "", None, now).unwrap();

        let top = state.chain(Some("TabList"));
        let Lookup::Ask(_waiting) = ask(&top) else {
            panic!("a top-level line asks");
        };
        let nested = state.chain(Some("Exp"));
        let held: Vec<_> = (0..MAX_NESTED_ASKS).map(|_| ask(&nested)).collect();
        assert!(held.iter().all(|lookup| matches!(lookup, Lookup::Ask(_))));
        let refused = ask(&nested);
        assert!(matches!(
            refused,
            Lookup::Refused {
                why: Refusal::Nesting,
                ..
            }
        ));
        assert!(matches!(
            ask(&state.chain(Some("Scoreboard"))),
            Lookup::Ask(_)
        ));
        drop(held);
        assert!(matches!(ask(&nested), Lookup::Ask(_)));
    }
}
