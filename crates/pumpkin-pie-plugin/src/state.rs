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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, TryLockError};
use std::time::{Duration, Instant};

use pumpkin_pie::expansion::{Expansion, Registry};
use pumpkin_pie::protocol::{Cache, MAX_MESSAGE_BYTES, ProtocolError};

/// Asks that may wait on an expansion at once.
///
/// The v0.1 API runs one call chain at a time, so this is how deeply an
/// expansion that calls the provider back can nest. Without a limit, an
/// expansion whose answer resolves its own placeholder recursed until Pumpkin
/// stopped the chain at 64 nested calls by trapping the plugin that made the
/// next one.
// One counter for the whole plugin: under the concurrent v0.2 API it would also
// cap asks from unrelated calls, so count per call chain if that ever bites.
pub(crate) const MAX_ASKS_IN_FLIGHT: usize = 4;

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
    /// Asks waiting on an expansion right now. Kept outside the lock because an
    /// ask holds its count across the IPC call; a counter never blocks.
    asks: AtomicUsize,
}

#[derive(Default)]
struct Inner {
    registry: Registry,
    /// Keyed by expansion source, then by viewer and placeholder.
    cache: ExpansionCache,
}

/// The state was already locked, which only a call coming back in while the
/// lock was held can cause.
#[derive(Debug)]
pub(crate) struct Busy;

/// What resolving an expansion placeholder needs next.
pub(crate) enum Lookup {
    /// No expansion owns the namespace.
    Unknown,
    /// A value the expansion let the provider reuse.
    Cached(Option<String>),
    /// Ask the expansion, then hand the answer to [`State::store`].
    Ask(Pending),
}

/// An ask about to be made, and what its answer may be cached against.
pub(crate) struct Pending {
    pub(crate) expansion: Expansion,
    key: CacheKey,
}

/// One ask in flight, counted until it is dropped.
pub(crate) struct InFlight<'a>(&'a AtomicUsize);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
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
            inner.registry.add(namespace, sender, names, cache).cloned()
        })
    }

    /// Forgets every namespace `sender` claimed and what was cached for it.
    pub(crate) fn unregister(&self, sender: &str) -> Result<Vec<String>, Busy> {
        self.with(|inner| {
            inner.cache.retain(|key, _| key.0 != sender);
            inner.registry.drop_source(sender)
        })
    }

    /// Every registered expansion, sorted by namespace.
    pub(crate) fn expansions(&self) -> Result<Vec<Expansion>, Busy> {
        self.with(|inner| inner.registry.all().cloned().collect())
    }

    /// Finds who answers `id` for `viewer`, and whether a cached value will do.
    pub(crate) fn lookup(
        &self,
        id: &str,
        viewer: &str,
        argument: Option<&str>,
        now: Instant,
    ) -> Result<Lookup, Busy> {
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
            Lookup::Ask(Pending { expansion, key })
        })
    }

    /// Counts an ask in flight, or refuses when [`MAX_ASKS_IN_FLIGHT`] already are.
    pub(crate) fn begin_ask(&self) -> Option<InFlight<'_>> {
        if self.asks.fetch_add(1, Ordering::Relaxed) >= MAX_ASKS_IN_FLIGHT {
            self.asks.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(InFlight(&self.asks))
    }

    /// Caches an answer if the expansion asked for a TTL.
    pub(crate) fn store(
        &self,
        pending: Pending,
        value: Option<String>,
        now: Instant,
    ) -> Result<(), Busy> {
        let ttl = pending.expansion.ttl_ms();
        self.with(|inner| {
            if ttl == 0 {
                return;
            }
            cache_insert(
                &mut inner.cache,
                pending.key,
                value,
                Duration::from_millis(ttl),
                now,
                MAX_CACHE_ENTRIES,
                MAX_CACHE_BYTES,
            );
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

    #[test]
    fn asks_in_flight_are_capped_and_released() {
        let state = State::default();
        let asks: Vec<_> = (0..MAX_ASKS_IN_FLIGHT)
            .map(|_| state.begin_ask().expect("below the cap"))
            .collect();
        assert!(state.begin_ask().is_none());
        drop(asks);
        assert!(state.begin_ask().is_some());
    }
}
