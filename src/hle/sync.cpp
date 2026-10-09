#include "host/frame_stats.h"
#include "hle/guest_fs.h"
#include "hle/sync.h"
#include "core/futex.h"
#include "log.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <time.h>
#endif
#if defined(__APPLE__)
#include "core/mac_vm.h"
#endif

namespace gsync {

namespace {

// ---------------------------------------------------------------- arena
// Objects must sit below 2^40 (DLAdaptiveMutex masks owners to 40 bits) and
// must be recognisable so a garbage slot is rejected instead of dereferenced.
constexpr std::uintptr_t kArenaBase = 0xE000000000ull;
constexpr std::size_t kArenaSize = 64u << 20;
constexpr std::size_t kGranule = 16;
constexpr std::size_t kClasses = 64;  // up to 1 KiB objects

std::mutex g_arena_mu;
std::uint8_t* g_arena = nullptr;
std::size_t g_arena_used = 0;
void* g_free[kClasses] = {};

std::uint8_t* arena_map() {
#if defined(_WIN32)
    return static_cast<std::uint8_t*>(
        VirtualAlloc(reinterpret_cast<void*>(kArenaBase), kArenaSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#elif defined(__APPLE__)
    // In bbhost's high window (core/mac_vm.h), or anywhere free in a test.
    return static_cast<std::uint8_t*>(mac_vm_map(reinterpret_cast<void*>(kArenaBase), kArenaSize, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0, MacVmPlace::NoReplace));
#else
    void* m = mmap(reinterpret_cast<void*>(kArenaBase), kArenaSize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (m == MAP_FAILED) {
        return nullptr;
    }
    if (reinterpret_cast<std::uintptr_t>(m) != kArenaBase) {
        munmap(m, kArenaSize);
        return nullptr;
    }
    return static_cast<std::uint8_t*>(m);
#endif
}

}  // namespace

void* arena_alloc(std::size_t n) {
    const std::size_t cls = (n + kGranule - 1) / kGranule;
    if (cls == 0 || cls > kClasses) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_arena_mu);
    if (!g_arena) {
        g_arena = arena_map();
        if (!g_arena) {
            host_log("sync arena mmap at 0x%llx failed", static_cast<unsigned long long>(kArenaBase));
            return nullptr;
        }
    }
    if (g_free[cls - 1]) {
        void* p = g_free[cls - 1];
        g_free[cls - 1] = *static_cast<void**>(p);
        std::memset(p, 0, cls * kGranule);
        return p;
    }
    const std::size_t sz = cls * kGranule;
    if (g_arena_used + sz > kArenaSize) {
        host_log("sync arena exhausted");
        return nullptr;
    }
    void* p = g_arena + g_arena_used;
    g_arena_used += sz;
    return p;
}

void arena_free(void* p, std::size_t n) {
    if (!p) {
        return;
    }
    const std::size_t cls = (n + kGranule - 1) / kGranule;
    std::lock_guard<std::mutex> lock(g_arena_mu);
    std::memset(p, 0, cls * kGranule);
    *static_cast<void**>(p) = g_free[cls - 1];
    g_free[cls - 1] = p;
}

bool arena_owns(const void* p) {
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    return a >= kArenaBase && a < kArenaBase + kArenaSize;
}

namespace {

constexpr std::uint32_t kMagicMutex = 0x5854554du;   // 'MUTX'
constexpr std::uint32_t kMagicCond = 0x444e4f43u;    // 'COND'
constexpr std::uint32_t kMagicRw = 0x4b4c5752u;      // 'RWLK'
constexpr std::uint32_t kMagicMattr = 0x5254414du;   // 'MATR'
constexpr std::uint32_t kMagicDead = 0x44414544u;    // 'DEAD'

// The lock word is the mutex: 0 free, 1 held, 2 held and a thread may be
// asleep on cv. Taking a free mutex and releasing one nobody waits for only
// swap it, which the guest does without the HLE thunk (mutex_lock_fast,
// mutex_unlock_fast); a contender takes m, marks the word 2 and sleeps on cv,
// and whoever releases a 2 wakes one sleeper under m, so no wakeup is lost.
// owner and count belong to the holder.
//
// m and cv (and every other object's here) are core/futex.h's word lock and
// sequence-word condition variable, not std::mutex/std::condition_variable:
// on Windows those are winpthreads', a few system calls and kernel objects
// a wait, millisecond timeouts against the system tick. A contended guest
// lock, a condition wait and a read/write sleeper now cost one
// WaitOnAddress, and their wake one WakeByAddress (BBHOST_FUTEX=0: the old
// primitives underneath, for an A/B).
struct Mutex {
    std::uint32_t magic;
    int type;
    std::atomic<std::uint32_t> word;
    int count;
    std::atomic<Thread*> owner;
    HostLock m;
    HostCondVar cv;
    int waiters;  // under m
    bool dead;    // under m
};

// Signals are counted so notify_one can never be absorbed by a waiter that
// was already satisfied (the seq-compare scheme lost wakeups).
struct Cond {
    std::uint32_t magic;
    HostLock m;
    HostCondVar cv;
    int signals;
    int waiters;
    bool dead;
};

// A reader-preferring rwlock whose state is one word: bit 31 a writer holds
// it, the low bits how many readers do. A reader takes and gives back its
// share with one compare-and-swap - without the thunk (rw_rdlock_fast,
// rw_unlock_fast); m and cv are only for sleeping: the character threads
// make ~800 rdlock/unlock calls a frame, and every one took m.
struct Rwlock {
    std::uint32_t magic;
    HostLock m;
    HostCondVar cv;
    std::atomic<std::uint32_t> state;
    std::atomic<std::uint32_t> waiters;  // threads asleep (or about to be) on cv
    std::atomic<Thread*> writer;         // the holder while bit 31 is set
    bool dead;
};
constexpr std::uint32_t kRwWriter = 0x80000000u;

struct MutexAttr {
    std::uint32_t magic;
    int type;
};

template <typename T>
T* make(std::uint32_t magic) {
    void* p = arena_alloc(sizeof(T));
    if (!p) {
        return nullptr;
    }
    T* t = new (p) T();
    t->magic = magic;
    return t;
}

template <typename T>
void destroy(T* t) {
    t->magic = kMagicDead;
    t->~T();
    arena_free(t, sizeof(T));
}

template <typename T>
T* get(void** slot, std::uint32_t magic) {
    if (!slot) {
        return nullptr;
    }
    void* p = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (!p || !arena_owns(p)) {
        return nullptr;
    }
    T* t = static_cast<T*>(p);
    return t->magic == magic ? t : nullptr;
}

// Lazily initialise a NULL slot (FreeBSD static initializer). Two threads may
// race; the loser frees its object.
template <typename T, typename Init>
T* get_or_init(void** slot, std::uint32_t magic, Init init) {
    if (!slot) {
        return nullptr;
    }
    T* t = get<T>(slot, magic);
    if (t) {
        return t;
    }
    void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (cur != nullptr) {
        return nullptr;  // garbage
    }
    T* fresh = make<T>(magic);
    if (!fresh) {
        return nullptr;
    }
    init(fresh);
    void* expected = nullptr;
    if (__atomic_compare_exchange_n(slot, &expected, static_cast<void*>(fresh), false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
        return fresh;
    }
    destroy(fresh);
    return get<T>(slot, magic);
}

template <typename Pred>
bool wait_until(HostCondVar& cv, std::unique_lock<HostLock>& lk, const Deadline* dl, Pred pred) {
    if (!dl) {
        cv.wait(lk, pred);
        return true;
    }
    return cv.wait_until(lk, *dl, pred);
}

thread_local Thread* t_current = nullptr;

// For the 300-flip report's timing line (counts(), below): how often the
// slow paths run, i.e. how much the sleeping primitive matters.
std::atomic<std::uint64_t> g_mutex_contended{0}, g_cond_waits{0}, g_cond_signals{0}, g_rw_sleeps{0};

}  // namespace

SyncCounts counts() {
    return {g_mutex_contended.load(std::memory_order_relaxed), g_cond_waits.load(std::memory_order_relaxed),
            g_cond_signals.load(std::memory_order_relaxed), g_rw_sleeps.load(std::memory_order_relaxed)};
}

// ---------------------------------------------------------------- thread
struct Thread {
    std::uint32_t magic;
    std::uint32_t pad;
    std::uint8_t body[248];  // owned by pthread.cpp (HostThreadBody)
};
static_assert(sizeof(Thread) == 256, "Thread is one arena granule set");

Thread* thread_new() {
    void* p = arena_alloc(sizeof(Thread));
    if (!p) {
        return nullptr;
    }
    auto* t = static_cast<Thread*>(p);
    t->magic = 0x44524854u;  // 'THRD'
    return t;
}

void thread_free(Thread* t) {
    if (t) {
        arena_free(t, sizeof(Thread));
    }
}

Thread* current_thread() {
    if (!t_current) {
        t_current = thread_new();
    }
    if (auto* tcb = static_cast<GuestTcb*>(t_guest_fs); tcb && tcb->sync_thread != t_current) {
        tcb->sync_thread = t_current;  // for the guest's raw mutex calls
    }
    return t_current;
}

void set_current_thread(Thread* t) { t_current = t; }

// ---------------------------------------------------------------- mutex
int mutex_init(void** slot, int type) {
    if (!slot) {
        return kEINVAL;
    }
    if (type < kMutexErrorCheck || type > kMutexAdaptive) {
        type = kMutexDefault;
    }
    Mutex* m = make<Mutex>(kMagicMutex);
    if (!m) {
        return kEAGAIN;
    }
    m->type = type;
    __atomic_store_n(slot, static_cast<void*>(m), __ATOMIC_RELEASE);
    return 0;
}

int mutex_destroy(void** slot) {
    Mutex* m = get<Mutex>(slot, kMagicMutex);
    if (!m) {
        return kEINVAL;
    }
    {
        std::unique_lock<HostLock> lk(m->m);
        if (m->word.load(std::memory_order_relaxed) != 0 || m->waiters) {
            return kEBUSY;
        }
        m->dead = true;
    }
    __atomic_store_n(slot, static_cast<void*>(nullptr), __ATOMIC_RELEASE);
    destroy(m);
    return 0;
}

namespace {

Mutex* mutex_get(void** slot) {
    return get_or_init<Mutex>(slot, kMagicMutex, [](Mutex* m) { m->type = kMutexDefault; });
}

// Holding it, the caller's own word swap: 0 -> 1.
bool mutex_take(Mutex* m, Thread* self) {
    std::uint32_t free_word = 0;
    if (!m->word.compare_exchange_strong(free_word, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
        return false;
    }
    m->owner.store(self, std::memory_order_relaxed);
    m->count = 1;
    return true;
}

// The holder lets go completely; a 2 means someone may be asleep.
void mutex_release(Mutex* m) {
    m->count = 0;
    m->owner.store(nullptr, std::memory_order_relaxed);
    if (m->word.exchange(0, std::memory_order_release) == 2) {
        std::lock_guard<HostLock> lk(m->m);
        m->cv.notify_one();
    }
}

int mutex_lock_impl(Mutex* m, const Deadline* dl, bool try_only) {
    Thread* self = current_thread();
    if (m->owner.load(std::memory_order_relaxed) == self) {
        if (m->type == kMutexRecursive) {
            ++m->count;
            return 0;
        }
        if (m->type == kMutexErrorCheck || try_only) {
            return try_only ? kEBUSY : kEDEADLK;
        }
        // NORMAL/ADAPTIVE: relocking is a deadlock on the console. Report
        // it instead of hanging the host.
        host_log("HLE: self-deadlock on normal mutex %p", static_cast<void*>(m));
        return kEDEADLK;
    }
    if (mutex_take(m, self)) {
        return 0;
    }
    if (try_only) {
        return kEBUSY;
    }
    // BBHOST_FUTEX_SPIN=<n>: look for the holder to let go before sleeping
    // (off by default: Kyo's stress test of the same lock measured a spin
    // costing 2-5x, and on the APU a spinning core takes the GPU's power).
    for (unsigned i = 0, n = host_futex_spin(); i < n; ++i) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#endif
        if (m->word.load(std::memory_order_relaxed) == 0 && mutex_take(m, self)) return 0;
    }
    g_mutex_contended.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock<HostLock> lk(m->m);
    ++m->waiters;
    // Marking the word 2 and finding it was 0 takes the mutex; otherwise the
    // holder will see the 2 and wake a sleeper.
    int r = 0;
    while (m->word.exchange(2, std::memory_order_acquire) != 0) {
        if (m->dead) {
            r = kEINVAL;
            break;
        }
        if (!dl) {
            m->cv.wait(lk);
        } else if (m->cv.wait_until(lk, *dl) == std::cv_status::timeout) {
            if (m->word.exchange(2, std::memory_order_acquire) != 0) {
                r = kETIMEDOUT;
            }
            break;
        }
    }
    --m->waiters;
    if (r != 0) {
        return r;
    }
    // Nobody else asleep (they would have counted themselves under m, which
    // this thread holds): back to 1, so the release needs no wake.
    if (m->waiters == 0) {
        m->word.store(1, std::memory_order_relaxed);
    }
    m->owner.store(self, std::memory_order_relaxed);
    m->count = 1;
    return 0;
}

}  // namespace

int mutex_lock(void** slot, const Deadline* dl) {
    MainThreadWait timed(0);
    Mutex* m = mutex_get(slot);
    if (!m) {
        return kEINVAL;
    }
    return mutex_lock_impl(m, dl, false);
}

int mutex_trylock(void** slot) {
    Mutex* m = mutex_get(slot);
    if (!m) {
        return kEINVAL;
    }
    return mutex_lock_impl(m, nullptr, true);
}

int mutex_unlock(void** slot) {
    Mutex* m = get<Mutex>(slot, kMagicMutex);
    if (!m) {
        return kEINVAL;
    }
    Thread* self = current_thread();
    if (m->owner.load(std::memory_order_relaxed) != self) {
        if (m->type == kMutexNormal || m->type == kMutexAdaptive) {
            // Normal mutexes may be unlocked by a non-owner on FreeBSD only
            // by accident; be permissive if nobody owns it.
            if (m->word.load(std::memory_order_relaxed) == 0) {
                return 0;
            }
        }
        return kEPERM;
    }
    if (--m->count > 0) {
        return 0;
    }
    mutex_release(m);
    return 0;
}

bool mutex_lock_fast(void** slot, Thread* self) {
    Mutex* m = get<Mutex>(slot, kMagicMutex);
    if (!m || !self) {
        return false;
    }
    if (m->owner.load(std::memory_order_relaxed) == self) {
        if (m->type != kMutexRecursive) {
            return false;  // the error (or the deadlock report) is the slow path's
        }
        ++m->count;
        return true;
    }
    return mutex_take(m, self);
}

bool mutex_unlock_fast(void** slot, Thread* self) {
    Mutex* m = get<Mutex>(slot, kMagicMutex);
    if (!m || !self || m->owner.load(std::memory_order_relaxed) != self) {
        return false;
    }
    if (m->count > 1) {
        --m->count;  // one level of a recursive hold
        return true;
    }
    m->count = 0;
    m->owner.store(nullptr, std::memory_order_relaxed);
    std::uint32_t held = 1;
    if (m->word.compare_exchange_strong(held, 0, std::memory_order_release, std::memory_order_relaxed)) {
        return true;
    }
    // A sleeper to wake: still held, so put it back for the slow path.
    m->owner.store(self, std::memory_order_relaxed);
    m->count = 1;
    return false;
}

// ---------------------------------------------------------------- mutexattr
int mutexattr_init(void** slot) {
    if (!slot) {
        return kEINVAL;
    }
    MutexAttr* a = make<MutexAttr>(kMagicMattr);
    if (!a) {
        return kEAGAIN;
    }
    a->type = kMutexDefault;
    *slot = a;
    return 0;
}

int mutexattr_destroy(void** slot) {
    MutexAttr* a = get<MutexAttr>(slot, kMagicMattr);
    if (!a) {
        return kEINVAL;
    }
    *slot = nullptr;
    destroy(a);
    return 0;
}

int mutexattr_settype(void** slot, int type) {
    MutexAttr* a = get<MutexAttr>(slot, kMagicMattr);
    if (!a || type < kMutexErrorCheck || type > kMutexAdaptive) {
        return kEINVAL;
    }
    a->type = type;
    return 0;
}

int mutexattr_gettype(void** slot, int* type) {
    MutexAttr* a = get<MutexAttr>(slot, kMagicMattr);
    if (!a || !type) {
        return kEINVAL;
    }
    *type = a->type;
    return 0;
}

int mutexattr_type(void** slot) {
    MutexAttr* a = get<MutexAttr>(slot, kMagicMattr);
    return a ? a->type : kMutexDefault;
}

// ---------------------------------------------------------------- cond
int cond_init(void** slot) {
    if (!slot) {
        return kEINVAL;
    }
    Cond* c = make<Cond>(kMagicCond);
    if (!c) {
        return kEAGAIN;
    }
    __atomic_store_n(slot, static_cast<void*>(c), __ATOMIC_RELEASE);
    return 0;
}

int cond_destroy(void** slot) {
    Cond* c = get<Cond>(slot, kMagicCond);
    if (!c) {
        return kEINVAL;
    }
    {
        std::unique_lock<HostLock> lk(c->m);
        if (c->waiters) {
            return kEBUSY;
        }
        c->dead = true;
    }
    __atomic_store_n(slot, static_cast<void*>(nullptr), __ATOMIC_RELEASE);
    destroy(c);
    return 0;
}

int cond_wait(void** cond_slot, void** mutex_slot, const Deadline* dl) {
    MainThreadWait timed(0);
    Cond* c = get_or_init<Cond>(cond_slot, kMagicCond, [](Cond*) {});
    Mutex* m = mutex_get(mutex_slot);
    if (!c || !m) {
        return kEINVAL;
    }
    Thread* self = current_thread();
    int saved_count = 0;
    std::unique_lock<HostLock> ck(c->m);
    ++c->waiters;
    {
        // Release the guest mutex completely (even a recursive one).
        if (m->owner.load(std::memory_order_relaxed) != self) {
            --c->waiters;
            return kEPERM;
        }
        saved_count = m->count;
        mutex_release(m);
    }
    g_cond_waits.fetch_add(1, std::memory_order_relaxed);
    bool ok = wait_until(c->cv, ck, dl, [&] { return c->signals > 0 || c->dead; });
    if (ok && c->signals > 0) {
        --c->signals;
    }
    --c->waiters;
    const bool dead = c->dead;
    ck.unlock();
    // Re-acquire the guest mutex with the saved recursion depth.
    int r = mutex_lock_impl(m, nullptr, false);
    if (r == 0) {
        m->count = saved_count;
    }
    if (dead) {
        return kEINVAL;
    }
    if (!ok) {
        return kETIMEDOUT;
    }
    return r;
}

int cond_signal(void** slot, bool all) {
    Cond* c = get_or_init<Cond>(slot, kMagicCond, [](Cond*) {});
    if (!c) {
        return kEINVAL;
    }
    std::unique_lock<HostLock> lk(c->m);
    if (!c->waiters) {
        return 0;
    }
    g_cond_signals.fetch_add(1, std::memory_order_relaxed);
    if (all) {
        c->signals = c->waiters;
        c->cv.notify_all();
    } else {
        if (c->signals < c->waiters) {
            ++c->signals;
        }
        c->cv.notify_one();
    }
    return 0;
}

// ---------------------------------------------------------------- rwlock
int rw_init(void** slot) {
    if (!slot) {
        return kEINVAL;
    }
    Rwlock* r = make<Rwlock>(kMagicRw);
    if (!r) {
        return kEAGAIN;
    }
    __atomic_store_n(slot, static_cast<void*>(r), __ATOMIC_RELEASE);
    return 0;
}

int rw_destroy(void** slot) {
    Rwlock* r = get<Rwlock>(slot, kMagicRw);
    if (!r) {
        return kEINVAL;
    }
    {
        std::unique_lock<HostLock> lk(r->m);
        if (r->state.load() != 0 || r->waiters.load() != 0) {
            return kEBUSY;
        }
        r->dead = true;
    }
    __atomic_store_n(slot, static_cast<void*>(nullptr), __ATOMIC_RELEASE);
    destroy(r);
    return 0;
}

namespace {

Rwlock* rw_get(void** slot) {
    return get_or_init<Rwlock>(slot, kMagicRw, [](Rwlock*) {});
}

// One attempt: a reader while no writer holds it, a writer when nobody does.
bool rw_try_take(Rwlock* r, bool write) {
    std::uint32_t s = r->state.load(std::memory_order_relaxed);
    if (write) {
        return s == 0 && r->state.compare_exchange_strong(s, kRwWriter, std::memory_order_acquire, std::memory_order_relaxed);
    }
    while (!(s & kRwWriter)) {
        if (r->state.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) return true;
    }
    return false;
}

// After the state changed: a sleeper re-checks under m (it counted itself in
// `waiters` before it looked at the state, so one of the two sees the other).
void rw_wake(Rwlock* r) {
    if (r->waiters.load() == 0) return;
    { std::lock_guard<HostLock> lk(r->m); }
    r->cv.notify_all();
}

int rw_lock_impl(Rwlock* r, bool write, const Deadline* dl, bool try_only) {
    Thread* self = current_thread();
    if (r->writer.load(std::memory_order_relaxed) == self && (r->state.load() & kRwWriter)) {
        return kEDEADLK;
    }
    if (!rw_try_take(r, write)) {
        if (try_only) {
            return kEBUSY;
        }
        g_rw_sleeps.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<HostLock> lk(r->m);
        r->waiters.fetch_add(1);
        const bool ok = wait_until(r->cv, lk, dl, [&] { return r->dead || rw_try_take(r, write); });
        r->waiters.fetch_sub(1);
        if (r->dead) {
            return kEINVAL;
        }
        if (!ok) {
            return kETIMEDOUT;
        }
    }
    if (write) r->writer.store(self, std::memory_order_relaxed);
    return 0;
}

}  // namespace

int rw_rdlock(void** slot, const Deadline* dl) {
    Rwlock* r = rw_get(slot);
    return r ? rw_lock_impl(r, false, dl, false) : kEINVAL;
}
int rw_wrlock(void** slot, const Deadline* dl) {
    Rwlock* r = rw_get(slot);
    return r ? rw_lock_impl(r, true, dl, false) : kEINVAL;
}
int rw_tryrdlock(void** slot) {
    Rwlock* r = rw_get(slot);
    return r ? rw_lock_impl(r, false, nullptr, true) : kEINVAL;
}
int rw_trywrlock(void** slot) {
    Rwlock* r = rw_get(slot);
    return r ? rw_lock_impl(r, true, nullptr, true) : kEINVAL;
}

int rw_unlock(void** slot) {
    Rwlock* r = get<Rwlock>(slot, kMagicRw);
    if (!r) {
        return kEINVAL;
    }
    std::uint32_t s = r->state.load(std::memory_order_relaxed);
    if (s & kRwWriter) {
        if (r->writer.load(std::memory_order_relaxed) != current_thread()) {
            return kEPERM;
        }
        r->writer.store(nullptr, std::memory_order_relaxed);
        r->state.fetch_and(~kRwWriter, std::memory_order_release);
    } else {
        do {
            if (s == 0 || (s & kRwWriter)) return kEPERM;
        } while (!r->state.compare_exchange_weak(s, s - 1, std::memory_order_release, std::memory_order_relaxed));
    }
    rw_wake(r);
    return 0;
}

bool rw_rdlock_fast(void** slot) {
    Rwlock* r = get<Rwlock>(slot, kMagicRw);
    return r && rw_try_take(r, false);
}

int rw_unlock_fast(void** slot) {
    Rwlock* r = get<Rwlock>(slot, kMagicRw);
    if (!r) return -1;
    std::uint32_t s = r->state.load(std::memory_order_relaxed);
    do {
        if (s == 0 || (s & kRwWriter)) return -1;  // a writer's unlock, or none held: the slow path's
    } while (!r->state.compare_exchange_weak(s, s - 1, std::memory_order_release, std::memory_order_relaxed));
    return r->waiters.load() != 0 ? 1 : 0;
}

void rw_wake_slot(void** slot) {
    if (Rwlock* r = get<Rwlock>(slot, kMagicRw)) rw_wake(r);
}

// ---------------------------------------------------------------- time
Deadline from_realtime_abs(std::int64_t sec, std::int64_t nsec) {
#if defined(_WIN32)
    const auto now_sys = std::chrono::system_clock::now();
#else
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    const auto now_sys = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec)));
#endif
    const auto target = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::seconds(sec) + std::chrono::nanoseconds(nsec)));
    auto delta = target - now_sys;
    if (delta.count() < 0) {
        delta = decltype(delta)::zero();
    }
    return Clock::now() + std::chrono::duration_cast<Clock::duration>(delta);
}

}  // namespace gsync
