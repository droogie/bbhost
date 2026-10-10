// Orbis pthread / libc pthread HLE over hle/sync.cpp. Handles are pointers
// written into the guest slot; see sync.h for the model.
#include "hle/common.h"
#include "hle/guest_fs.h"
#include "hle/hle.h"
#include "hle/platform.h"
#include "hle/sync.h"
#include "core/portable.h"
#include "core/thunk.h"
#include "core/tls_rewrite.h"

#include <pthread.h>  // winpthreads on Windows: the same API

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace {

using gsync::Deadline;

// Body of gsync::Thread (248 bytes available).
struct ThreadBody {
    pthread_t pt;
    std::uint32_t tid;
    int prio;
    char name[32];
    void* entry;
    void* arg;
    void* retval;
    void* tcb;
    std::uint64_t exit_jb[12];
    bool can_exit;
    bool detached;
    bool started;
    bool host_born;
    bool finished;
};
static_assert(sizeof(ThreadBody) <= 248, "ThreadBody fits gsync::Thread");

// Publish the native handle before a child can run or release its body.
// Completion and detach also share this lock: either may be the last user
// of a detached handle, including a detach after the thread has returned.
std::mutex g_thread_lifecycle_mu;

ThreadBody* body(gsync::Thread* t) {
    return reinterpret_cast<ThreadBody*>(reinterpret_cast<std::uint8_t*>(t) + 8);
}

gsync::Thread* self_thread() {
    gsync::Thread* t = gsync::current_thread();
    ThreadBody* b = body(t);
    if (!b->started) {
        // Host-born thread (boot thread or a host helper) seen for the first time.
        b->started = true;
        b->host_born = true;
        b->prio = 700;
        b->pt = pthread_self();
        b->tid = host_thread_id();
    }
    if (auto* tcb = static_cast<GuestTcb*>(t_guest_fs); tcb && tcb->pthread_self != t) {
        tcb->pthread_self = t;  // for the guest's raw scePthreadSelf
    }
    return t;
}

// FreeBSD timespec.
struct BsdTimespec {
    std::int64_t sec;
    std::int64_t nsec;
};

int bsd(int e) { return e; }  // gsync:: already returns FreeBSD errno
int sce(int e) { return e == 0 ? 0 : static_cast<int>(0x80020000u | static_cast<unsigned>(e)); }

// ------------------------------------------------------------- mutex
GUEST_ABI int hle_mutex_init(void** m, void** attr) { return bsd(gsync::mutex_init(m, gsync::mutexattr_type(attr))); }
GUEST_ABI int hle_mutex_lock(void** m) { return bsd(gsync::mutex_lock(m, nullptr)); }
GUEST_ABI int hle_mutex_trylock(void** m) { return bsd(gsync::mutex_trylock(m)); }
GUEST_ABI int hle_mutex_unlock(void** m) { return bsd(gsync::mutex_unlock(m)); }
GUEST_ABI int hle_mutex_destroy(void** m) { return bsd(gsync::mutex_destroy(m)); }

GUEST_ABI int hle_sce_mutex_init(void** m, void** attr, const char*) {
    return sce(gsync::mutex_init(m, gsync::mutexattr_type(attr)));
}
GUEST_ABI int hle_sce_mutex_lock(void** m) { return sce(gsync::mutex_lock(m, nullptr)); }
GUEST_ABI int hle_sce_mutex_trylock(void** m) { return sce(gsync::mutex_trylock(m)); }
GUEST_ABI int hle_sce_mutex_unlock(void** m) { return sce(gsync::mutex_unlock(m)); }
GUEST_ABI int hle_sce_mutex_destroy(void** m) { return sce(gsync::mutex_destroy(m)); }
GUEST_ABI int hle_sce_mutex_timedlock(void** m, std::uint32_t usec) {
    Deadline dl = gsync::after_us(usec);
    return sce(gsync::mutex_lock(m, &dl));
}

// The entries the game calls most - on the main loop and the GX workers,
// about a third of all their HLE calls - run without the HLE thunk: guest FS,
// guest stack, so no host TLS. The thread comes from the guest TCB (fs:0x30 /
// fs:0x38, filled in by the thread's first thunked call), once fs:0x10 shows
// the TCB is ours. Everything but the uncontended case - a static
// initializer, a contended or recursive-error lock, a sleeper to wake, a
// thread that has made no thunked call yet - goes on to the thunked entry.
template <std::size_t Off>
gsync::Thread* tcb_thread() {
    if (tls_gs_mode()) {
        // The TCB from the GS slot the rewritten guest reads (core/tls_rewrite.h).
        static const std::uint32_t disp = tls_gs_disp();
        const GuestTcb* tcb = nullptr;
        asm volatile("movq %%gs:(%1), %0" : "=r"(tcb) : "r"(static_cast<std::uint64_t>(disp)));
        if (!tcb || tcb->magic != kTcbMagic) return nullptr;
        return *reinterpret_cast<gsync::Thread* const*>(reinterpret_cast<const std::uint8_t*>(tcb) + Off);
    }
    std::uint64_t magic = 0;
    asm volatile("movq %%fs:0x10, %0" : "=r"(magic));
    if (magic != kTcbMagic) {
        return nullptr;
    }
    gsync::Thread* t = nullptr;
    asm volatile("movq %%fs:%c1, %0" : "=r"(t) : "i"(Off));
    return t;
}
using MutexFn = int(GUEST_ABI*)(void**);
MutexFn g_slow_mutex_lock, g_slow_mutex_unlock, g_slow_sce_mutex_lock, g_slow_sce_mutex_unlock;
std::uint64_t(GUEST_ABI* g_slow_pthread_self)();

GUEST_ABI int raw_mutex_lock(void** m) {
    return gsync::mutex_lock_fast(m, tcb_thread<offsetof(GuestTcb, sync_thread)>()) ? 0 : g_slow_mutex_lock(m);
}
GUEST_ABI int raw_mutex_unlock(void** m) {
    return gsync::mutex_unlock_fast(m, tcb_thread<offsetof(GuestTcb, sync_thread)>()) ? 0 : g_slow_mutex_unlock(m);
}
GUEST_ABI int raw_sce_mutex_lock(void** m) {
    return gsync::mutex_lock_fast(m, tcb_thread<offsetof(GuestTcb, sync_thread)>()) ? 0 : g_slow_sce_mutex_lock(m);
}
GUEST_ABI int raw_sce_mutex_unlock(void** m) {
    return gsync::mutex_unlock_fast(m, tcb_thread<offsetof(GuestTcb, sync_thread)>()) ? 0 : g_slow_sce_mutex_unlock(m);
}
// The character threads' rwlocks (~800 read lock/unlock calls a frame): a
// reader's share in one compare-and-swap; anything else - a writer, a lock
// not made yet, sleepers to wake - goes on to the thunked entries.
using RwFn = int(GUEST_ABI*)(void**);
RwFn g_slow_sce_rw_rdlock, g_slow_sce_rw_unlock, g_slow_sce_rw_wake;
GUEST_ABI int hle_sce_rwlock_wake(void** r) {
    gsync::rw_wake_slot(r);
    return 0;
}
GUEST_ABI int raw_sce_rwlock_rdlock(void** r) { return gsync::rw_rdlock_fast(r) ? 0 : g_slow_sce_rw_rdlock(r); }
GUEST_ABI int raw_sce_rwlock_unlock(void** r) {
    const int done = gsync::rw_unlock_fast(r);
    if (done < 0) return g_slow_sce_rw_unlock(r);
    if (done > 0) g_slow_sce_rw_wake(r);
    return 0;
}
GUEST_ABI std::uint64_t raw_pthread_self() {
    const gsync::Thread* t = tcb_thread<offsetof(GuestTcb, pthread_self)>();
    return t ? reinterpret_cast<std::uint64_t>(t) : g_slow_pthread_self();
}

GUEST_ABI int hle_mutexattr_init(void** a) { return bsd(gsync::mutexattr_init(a)); }
GUEST_ABI int hle_mutexattr_destroy(void** a) { return bsd(gsync::mutexattr_destroy(a)); }
GUEST_ABI int hle_mutexattr_settype(void** a, int type) { return bsd(gsync::mutexattr_settype(a, type)); }
GUEST_ABI int hle_sce_mutexattr_init(void** a) { return sce(gsync::mutexattr_init(a)); }
GUEST_ABI int hle_sce_mutexattr_destroy(void** a) { return sce(gsync::mutexattr_destroy(a)); }
GUEST_ABI int hle_sce_mutexattr_settype(void** a, int type) { return sce(gsync::mutexattr_settype(a, type)); }

// ------------------------------------------------------------- cond
GUEST_ABI int hle_cond_init(void** c, void*) { return bsd(gsync::cond_init(c)); }
GUEST_ABI int hle_cond_destroy(void** c) { return bsd(gsync::cond_destroy(c)); }
GUEST_ABI int hle_cond_wait(void** c, void** m) { return bsd(gsync::cond_wait(c, m, nullptr)); }
GUEST_ABI int hle_cond_timedwait(void** c, void** m, const BsdTimespec* abs) {
    if (!abs) {
        return bsd(gsync::cond_wait(c, m, nullptr));
    }
    Deadline dl = gsync::from_realtime_abs(abs->sec, abs->nsec);
    return bsd(gsync::cond_wait(c, m, &dl));
}
GUEST_ABI int hle_cond_signal(void** c) { return bsd(gsync::cond_signal(c, false)); }
GUEST_ABI int hle_cond_broadcast(void** c) { return bsd(gsync::cond_signal(c, true)); }
GUEST_ABI int hle_sce_cond_init(void** c, void*, const char*) { return sce(gsync::cond_init(c)); }
GUEST_ABI int hle_sce_cond_destroy(void** c) { return sce(gsync::cond_destroy(c)); }
GUEST_ABI int hle_sce_cond_wait(void** c, void** m) { return sce(gsync::cond_wait(c, m, nullptr)); }
GUEST_ABI int hle_sce_cond_timedwait(void** c, void** m, std::uint32_t usec) {
    Deadline dl = gsync::after_us(usec);
    return sce(gsync::cond_wait(c, m, &dl));
}
GUEST_ABI int hle_sce_cond_signal(void** c) { return sce(gsync::cond_signal(c, false)); }
GUEST_ABI int hle_sce_cond_broadcast(void** c) { return sce(gsync::cond_signal(c, true)); }

// ------------------------------------------------------------- rwlock
GUEST_ABI int hle_sce_rwlock_init(void** r, void*, const char*) { return sce(gsync::rw_init(r)); }
GUEST_ABI int hle_sce_rwlock_destroy(void** r) { return sce(gsync::rw_destroy(r)); }
GUEST_ABI int hle_sce_rwlock_rdlock(void** r) { return sce(gsync::rw_rdlock(r, nullptr)); }
GUEST_ABI int hle_sce_rwlock_wrlock(void** r) { return sce(gsync::rw_wrlock(r, nullptr)); }
GUEST_ABI int hle_sce_rwlock_unlock(void** r) { return sce(gsync::rw_unlock(r)); }
GUEST_ABI int hle_sce_rwlock_tryrd(void** r) { return sce(gsync::rw_tryrdlock(r)); }
GUEST_ABI int hle_sce_rwlock_trywr(void** r) { return sce(gsync::rw_trywrlock(r)); }
GUEST_ABI int hle_sce_rwlock_timedrd(void** r, std::uint32_t usec) {
    Deadline dl = gsync::after_us(usec);
    return sce(gsync::rw_rdlock(r, &dl));
}
GUEST_ABI int hle_sce_rwlock_timedwr(void** r, std::uint32_t usec) {
    Deadline dl = gsync::after_us(usec);
    return sce(gsync::rw_wrlock(r, &dl));
}

// ------------------------------------------------------------- attr
struct PthreadAttr {
    std::uint32_t magic;
    int detach;
    std::size_t stacksize;
    int policy;
    int prio;
};
constexpr std::uint32_t kAttrMagic = 0x52545441u;  // 'ATTR'

PthreadAttr* attr_get(void** slot) {
    if (!slot) {
        return nullptr;
    }
    void* p = *slot;
    if (!p || !gsync::arena_owns(p)) {
        return nullptr;
    }
    auto* a = static_cast<PthreadAttr*>(p);
    return a->magic == kAttrMagic ? a : nullptr;
}

GUEST_ABI int hle_attr_init(void** a) {
    if (!a) {
        return gsync::kEINVAL;
    }
    auto* p = static_cast<PthreadAttr*>(gsync::arena_alloc(sizeof(PthreadAttr)));
    if (!p) {
        return gsync::kEAGAIN;
    }
    p->magic = kAttrMagic;
    p->prio = 700;
    *a = p;
    return 0;
}
GUEST_ABI int hle_attr_destroy(void** a) {
    PthreadAttr* p = attr_get(a);
    if (!p) {
        return gsync::kEINVAL;
    }
    p->magic = 0;
    gsync::arena_free(p, sizeof(PthreadAttr));
    *a = nullptr;
    return 0;
}
GUEST_ABI int hle_attr_setdetach(void** a, int v) {
    PthreadAttr* p = attr_get(a);
    if (!p) {
        return gsync::kEINVAL;
    }
    p->detach = v;
    return 0;
}
GUEST_ABI int hle_attr_setstack(void** a, std::uint64_t v) {
    PthreadAttr* p = attr_get(a);
    if (!p) {
        return gsync::kEINVAL;
    }
    p->stacksize = static_cast<std::size_t>(v);
    return 0;
}
GUEST_ABI int hle_attr_setsched(void** a, const int* param) {
    PthreadAttr* p = attr_get(a);
    if (!p) {
        return gsync::kEINVAL;
    }
    if (param) {
        p->prio = *param;
    }
    return 0;
}
GUEST_ABI int hle_attr_setpolicy(void** a, int v) {
    PthreadAttr* p = attr_get(a);
    if (!p) {
        return gsync::kEINVAL;
    }
    p->policy = v;
    return 0;
}
GUEST_ABI int hle_attr_setaffinity(void** a, void*) { return attr_get(a) ? 0 : gsync::kEINVAL; }
GUEST_ABI int hle_sce_attr_init(void** a) { return sce(hle_attr_init(a)); }
GUEST_ABI int hle_sce_attr_destroy(void** a) { return sce(hle_attr_destroy(a)); }
GUEST_ABI int hle_sce_attr_setdetach(void** a, int v) { return sce(hle_attr_setdetach(a, v)); }
GUEST_ABI int hle_sce_attr_setstack(void** a, std::uint64_t v) { return sce(hle_attr_setstack(a, v)); }
GUEST_ABI int hle_sce_attr_setsched(void** a, const int* v) { return sce(hle_attr_setsched(a, v)); }
GUEST_ABI int hle_sce_attr_setpolicy(void** a, int v) { return sce(hle_attr_setpolicy(a, v)); }
GUEST_ABI int hle_sce_attr_setaffinity(void** a, void* v) { return sce(hle_attr_setaffinity(a, v)); }

// ------------------------------------------------------------- threads
GUEST_ABI std::uint64_t hle_pthread_self() {
    return reinterpret_cast<std::uint64_t>(self_thread());
}
GUEST_ABI std::uint64_t hle_sce_pthread_self() { return hle_pthread_self(); }

GUEST_ABI int hle_pthread_setaffinity(void*, int) { return 0; }
GUEST_ABI int hle_sce_yield() {
#if defined(_WIN32)
    Sleep(0);
#else
    sched_yield();
#endif
    return 0;
}

// tls_index { module, offset }. The eboot is the only TLS module; its block
// is the guest thread's static TLS right below the TCB (variant II), the
// same bytes the fs:-relative accesses use.
GUEST_ABI void* hle_tls_get_addr(std::uint64_t* idx) {
    const std::uint64_t off = idx ? idx[1] : 0;
    auto* tcb = static_cast<std::uint8_t*>(t_guest_fs);
    if (!tcb) {
        static thread_local unsigned char fallback[kGuestTls];
        return fallback + (off < kGuestTls ? off : 0);
    }
    return tcb - kGuestTls + (off < kGuestTls ? off : 0);
}

// Guest threads made by scePthreadCreate, and those of them not yet in
// their entry: the movie player's READY waits on both (hle_threads_made,
// avplayer.cpp).
std::atomic<std::uint64_t> g_threads_made{0};
std::atomic<int> g_threads_starting{0};

// BBHOST_TEST_THREAD_START_MS=N: every guest thread waits N ms before its
// entry - a loaded machine's slow thread start, on demand.
int test_thread_start_ms() {
    static const int ms = [] {
        const char* e = std::getenv("BBHOST_TEST_THREAD_START_MS");
        return e ? std::atoi(e) : 0;
    }();
    return ms;
}

void* thread_tramp(void* p) {
    {
        std::lock_guard<std::mutex> lock(g_thread_lifecycle_mu);
    }
    auto* t = static_cast<gsync::Thread*>(p);
    ThreadBody* b = body(t);
    gsync::set_current_thread(t);
    b->tid = host_thread_id();
    b->started = true;
    if (b->name[0]) {
        host_thread_set_name(b->name);
    }
    using Fn = void*(GUEST_ABI*)(void*);  // the guest's thread function
    // scePthreadExit longjmps back here from inside HLE (host FS, host
    // stack); hle_setjmp_raw has no same-stack check.
    if (hle_setjmp_raw(b->exit_jb) == 0) {
        b->can_exit = true;
        void* tcb = guest_thread_enter();
        b->tcb = tcb;
        if (const int ms = test_thread_start_ms()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
        g_threads_starting.fetch_sub(1);
        b->retval = reinterpret_cast<Fn>(b->entry)(b->arg);
        b->can_exit = false;
        guest_thread_leave(tcb);
    } else {
        // Arrived via hle_pthread_exit. Host FS is installed already.
        b->can_exit = false;
        guest_thread_leave(b->tcb);
    }
    void* r = b->retval;
    {
        std::lock_guard<std::mutex> lock(g_thread_lifecycle_mu);
        b->finished = true;
        if (b->detached) {
            gsync::thread_free(t);
        }
    }
    return r;
}

int pthread_create_impl(void** th, void** attr, void* entry, void* arg, const char* name) {
    if (!th || !entry) {
        return gsync::kEINVAL;
    }
    PthreadAttr* ga = attr_get(attr);
    // PS4 workers often request 64KiB. FMOD's FEV parser uses a ~4KiB frame
    // and recurses, which blows those stacks on the host.
    constexpr std::size_t kMinGuestStack = 2 * 1024 * 1024;
    std::size_t stack = ga ? ga->stacksize : 0;
    if (stack && stack < kMinGuestStack) {
        stack = kMinGuestStack;
    }
    gsync::Thread* t = gsync::thread_new();
    if (!t) {
        return gsync::kEAGAIN;
    }
    ThreadBody* b = body(t);
    b->entry = entry;
    b->arg = arg;
    b->prio = ga ? ga->prio : 700;
    b->detached = ga && ga->detach;
    if (name) {
        std::strncpy(b->name, name, sizeof(b->name) - 1);
    }
    pthread_attr_t ha;
    pthread_attr_init(&ha);
    if (stack >= 16384) {
        pthread_attr_setstacksize(&ha, stack);
    }
    if (b->detached) {
        pthread_attr_setdetachstate(&ha, PTHREAD_CREATE_DETACHED);
    }
    static int logs = 0;
    if (logs < 48) {
        ++logs;
        host_log("pthread_create %s entry=0x%llx arg=%p stack=0x%zx -> %p", b->name,
                 static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(entry)), arg, stack,
                 static_cast<void*>(t));
    }
    // Publish the handle before the thread can run so the child never sees
    // an empty slot.
    std::lock_guard<std::mutex> lock(g_thread_lifecycle_mu);
    *th = t;
    g_threads_starting.fetch_add(1);
    g_threads_made.fetch_add(1);
    pthread_t pt;
    int e = pthread_create(&pt, &ha, thread_tramp, t);
    pthread_attr_destroy(&ha);
    if (e) {
        g_threads_starting.fetch_sub(1);
        *th = nullptr;
        gsync::thread_free(t);
        return to_freebsd(e);
    }
    // The child is still behind the lifecycle lock. Even an immediately
    // exiting detached thread cannot free b before this publication.
    b->pt = pt;
    return 0;
}

GUEST_ABI int hle_pthread_create(void** th, void** attr, void* entry, void* arg) {
    return pthread_create_impl(th, attr, entry, arg, nullptr);
}
GUEST_ABI int hle_pthread_create_name(void** th, void** attr, void* entry, void* arg, const char* name) {
    return pthread_create_impl(th, attr, entry, arg, name);
}
GUEST_ABI int hle_sce_pthread_create(void** th, void** attr, void* entry, void* arg, const char* name) {
    return sce(pthread_create_impl(th, attr, entry, arg, name));
}

int pthread_join_impl(void* th, void** ret) {
    gsync::Thread* t = th ? static_cast<gsync::Thread*>(th) : nullptr;
    if (!t || !gsync::arena_owns(t)) {
        return gsync::kEINVAL;
    }
    ThreadBody* b = body(t);
    pthread_t pt;
    {
        std::lock_guard<std::mutex> lock(g_thread_lifecycle_mu);
        if (b->host_born || b->detached) {
            return gsync::kEINVAL;
        }
        pt = b->pt;
    }
    void* r = nullptr;
    int e = pthread_join(pt, &r);
    if (e) {
        return to_freebsd(e);
    }
    if (ret) {
        *ret = r;
    }
    gsync::thread_free(t);
    return 0;
}
// pthread_join(pthread_t, void**): the handle itself, by value.
GUEST_ABI int hle_pthread_join(void* th, void** ret) { return pthread_join_impl(th, ret); }
GUEST_ABI int hle_sce_pthread_join(void* th, void** ret) { return sce(pthread_join_impl(th, ret)); }

int pthread_detach_impl(void* th) {
    gsync::Thread* t = static_cast<gsync::Thread*>(th);
    if (!t || !gsync::arena_owns(t)) {
        return gsync::kEINVAL;
    }
    ThreadBody* b = body(t);
    std::lock_guard<std::mutex> lock(g_thread_lifecycle_mu);
    if (b->host_born || b->detached) {
        return gsync::kEINVAL;
    }
    const int e = pthread_detach(b->pt);
    if (e) {
        return to_freebsd(e);
    }
    b->detached = true;
    if (b->finished) {
        gsync::thread_free(t);
    }
    return 0;
}
GUEST_ABI int hle_sce_pthread_detach(void* th) { return sce(pthread_detach_impl(th)); }

GUEST_ABI void hle_pthread_exit(void* r) {
    gsync::Thread* t = gsync::current_thread();
    ThreadBody* b = body(t);
    if (b->can_exit) {
        b->retval = r;
        hle_longjmp_raw(b->exit_jb, 1);
    }
    host_log("pthread_exit on a thread without a trampoline; exiting the host thread");
    pthread_exit(r);
}

GUEST_ABI int hle_sce_rename(void* th, const char* name) {
    gsync::Thread* t = static_cast<gsync::Thread*>(th);
    if (!t || !gsync::arena_owns(t) || !name) {
        return sce(gsync::kEINVAL);
    }
    ThreadBody* b = body(t);
    std::strncpy(b->name, name, sizeof(b->name) - 1);
    if (b->started) {
#if defined(_WIN32)
        pthread_setname_np(b->pt, b->name);
#else
        char comm[16];
        host_thread_fit_name(b->name, comm);
        pthread_setname_np(b->pt, comm);
#endif
    }
    return 0;
}
GUEST_ABI int hle_sce_getprio(void* th, int* p) {
    gsync::Thread* t = static_cast<gsync::Thread*>(th);
    if (!t || !gsync::arena_owns(t) || !p) {
        return sce(gsync::kEINVAL);
    }
    *p = body(t)->prio;
    return 0;
}
GUEST_ABI int hle_sce_setprio(void* th, int prio) {
    gsync::Thread* t = static_cast<gsync::Thread*>(th);
    if (!t || !gsync::arena_owns(t)) {
        return sce(gsync::kEINVAL);
    }
    body(t)->prio = prio;
    return 0;
}

// ------------------------------------------------------------- Dinkum / misc
std::recursive_mutex g_syslocks[32];
GUEST_ABI int hle_locksyslock(int id) {
    if (id >= 0 && id < 32) {
        g_syslocks[id].lock();
    }
    return 0;
}
GUEST_ABI int hle_unlocksyslock(int id) {
    if (id >= 0 && id < 32) {
        g_syslocks[id].unlock();
    }
    return 0;
}
// _Mtx_t is a pointer; flags: 0x100 recursive.
GUEST_ABI int hle_mtx_init(void** m, int flags) {
    return gsync::mutex_init(m, (flags & 0x100) ? gsync::kMutexRecursive : gsync::kMutexErrorCheck) ? 2 : 0;
}
GUEST_ABI int hle_mtx_destroy(void** m) {
    gsync::mutex_destroy(m);
    return 0;
}
GUEST_ABI int hle_mtx_lock(void** m) { return gsync::mutex_lock(m, nullptr) ? 2 : 0; }
GUEST_ABI int hle_mtx_trylock(void** m) {
    int r = gsync::mutex_trylock(m);
    return r == 0 ? 0 : (r == gsync::kEBUSY ? 3 : 2);  // _Thrd_busy = 3
}
GUEST_ABI int hle_mtx_unlock(void** m) { return gsync::mutex_unlock(m) ? 2 : 0; }

GUEST_ABI unsigned hle_atomic_add4(std::atomic<unsigned>* p, unsigned v) {
    return p ? p->fetch_add(v) : 0;
}
GUEST_ABI unsigned hle_atomic_sub4(std::atomic<unsigned>* p, unsigned v) {
    return p ? p->fetch_sub(v) : 0;
}

}  // namespace

std::uint64_t hle_threads_made() { return g_threads_made.load(); }
int hle_threads_starting() { return g_threads_starting.load(); }

void hle_register_pthread() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    // The raw entries (above) and the thunked ones they fall back on.
    g_slow_mutex_lock = reinterpret_cast<MutexFn>(thunk_wrap(reinterpret_cast<void*>(&hle_mutex_lock)));
    g_slow_mutex_unlock = reinterpret_cast<MutexFn>(thunk_wrap(reinterpret_cast<void*>(&hle_mutex_unlock)));
    g_slow_sce_mutex_lock = reinterpret_cast<MutexFn>(thunk_wrap(reinterpret_cast<void*>(&hle_sce_mutex_lock)));
    g_slow_sce_mutex_unlock = reinterpret_cast<MutexFn>(thunk_wrap(reinterpret_cast<void*>(&hle_sce_mutex_unlock)));
    g_slow_pthread_self = reinterpret_cast<std::uint64_t(GUEST_ABI*)()>(thunk_wrap(reinterpret_cast<void*>(&hle_pthread_self)));
    g_slow_sce_rw_rdlock = reinterpret_cast<RwFn>(thunk_wrap(reinterpret_cast<void*>(&hle_sce_rwlock_rdlock)));
    g_slow_sce_rw_unlock = reinterpret_cast<RwFn>(thunk_wrap(reinterpret_cast<void*>(&hle_sce_rwlock_unlock)));
    g_slow_sce_rw_wake = reinterpret_cast<RwFn>(thunk_wrap(reinterpret_cast<void*>(&hle_sce_rwlock_wake)));
    if (const char* e = std::getenv("BBHOST_RAW_SYNC"); !e || e[0] != '0') {
#define REG_RAW(name, fn) register_hle_fn_raw(name, reinterpret_cast<void*>(fn))
        REG_RAW("pthread_self", raw_pthread_self);
        REG_RAW("scePthreadSelf", raw_pthread_self);
        REG_RAW("pthread_mutex_lock", raw_mutex_lock);
        REG_RAW("pthread_mutex_unlock", raw_mutex_unlock);
        REG_RAW("scePthreadMutexLock", raw_sce_mutex_lock);
        REG_RAW("scePthreadMutexUnlock", raw_sce_mutex_unlock);
        REG_RAW("scePthreadRwlockRdlock", raw_sce_rwlock_rdlock);
        REG_RAW("scePthreadRwlockUnlock", raw_sce_rwlock_unlock);
#undef REG_RAW
    } else {
        REG("scePthreadRwlockRdlock", hle_sce_rwlock_rdlock);
        REG("scePthreadRwlockUnlock", hle_sce_rwlock_unlock);
        REG("pthread_self", hle_pthread_self);
        REG("scePthreadSelf", hle_sce_pthread_self);
        REG("pthread_mutex_lock", hle_mutex_lock);
        REG("pthread_mutex_unlock", hle_mutex_unlock);
        REG("scePthreadMutexLock", hle_sce_mutex_lock);
        REG("scePthreadMutexUnlock", hle_sce_mutex_unlock);
    }
    REG("scePthreadSetaffinity", hle_pthread_setaffinity);
    REG("scePthreadYield", hle_sce_yield);
    REG("__tls_get_addr", hle_tls_get_addr);
    REG("pthread_mutex_init", hle_mutex_init);
    REG("pthread_mutex_trylock", hle_mutex_trylock);
    REG("pthread_mutex_destroy", hle_mutex_destroy);
    REG("scePthreadMutexInit", hle_sce_mutex_init);
    REG("scePthreadMutexTrylock", hle_sce_mutex_trylock);
    REG("scePthreadMutexDestroy", hle_sce_mutex_destroy);
    REG("scePthreadMutexTimedlock", hle_sce_mutex_timedlock);
    REG("pthread_mutexattr_init", hle_mutexattr_init);
    REG("pthread_mutexattr_destroy", hle_mutexattr_destroy);
    REG("pthread_mutexattr_settype", hle_mutexattr_settype);
    REG("scePthreadMutexattrInit", hle_sce_mutexattr_init);
    REG("scePthreadMutexattrDestroy", hle_sce_mutexattr_destroy);
    REG("scePthreadMutexattrSettype", hle_sce_mutexattr_settype);
    REG("pthread_cond_init", hle_cond_init);
    REG("pthread_cond_destroy", hle_cond_destroy);
    REG("pthread_cond_wait", hle_cond_wait);
    REG("pthread_cond_timedwait", hle_cond_timedwait);
    REG("pthread_cond_signal", hle_cond_signal);
    REG("pthread_cond_broadcast", hle_cond_broadcast);
    REG("scePthreadCondInit", hle_sce_cond_init);
    REG("scePthreadCondDestroy", hle_sce_cond_destroy);
    REG("scePthreadCondWait", hle_sce_cond_wait);
    REG("scePthreadCondTimedwait", hle_sce_cond_timedwait);
    REG("scePthreadCondSignal", hle_sce_cond_signal);
    REG("scePthreadCondBroadcast", hle_sce_cond_broadcast);
    REG("scePthreadRwlockInit", hle_sce_rwlock_init);
    REG("scePthreadRwlockDestroy", hle_sce_rwlock_destroy);
    REG("scePthreadRwlockWrlock", hle_sce_rwlock_wrlock);
    REG("scePthreadRwlockTryrdlock", hle_sce_rwlock_tryrd);
    REG("scePthreadRwlockTrywrlock", hle_sce_rwlock_trywr);
    REG("scePthreadRwlockTimedrdlock", hle_sce_rwlock_timedrd);
    REG("scePthreadRwlockTimedwrlock", hle_sce_rwlock_timedwr);
    REG("pthread_attr_init", hle_attr_init);
    REG("pthread_attr_destroy", hle_attr_destroy);
    REG("pthread_attr_setdetachstate", hle_attr_setdetach);
    REG("pthread_attr_setstacksize", hle_attr_setstack);
    REG("pthread_attr_setschedparam", hle_attr_setsched);
    REG("scePthreadAttrInit", hle_sce_attr_init);
    REG("scePthreadAttrDestroy", hle_sce_attr_destroy);
    REG("scePthreadAttrSetdetachstate", hle_sce_attr_setdetach);
    REG("scePthreadAttrSetstacksize", hle_sce_attr_setstack);
    REG("scePthreadAttrSetschedparam", hle_sce_attr_setsched);
    REG("scePthreadAttrSetschedpolicy", hle_sce_attr_setpolicy);
    REG("scePthreadAttrSetaffinity", hle_sce_attr_setaffinity);
    REG("pthread_create", hle_pthread_create);
    REG("pthread_create_name_np", hle_pthread_create_name);
    REG("scePthreadCreate", hle_sce_pthread_create);
    REG("pthread_join", hle_pthread_join);
    REG("scePthreadJoin", hle_sce_pthread_join);
    REG("scePthreadDetach", hle_sce_pthread_detach);
    REG("pthread_exit", hle_pthread_exit);
    REG("scePthreadExit", hle_pthread_exit);
    REG("scePthreadRename", hle_sce_rename);
    REG("scePthreadGetprio", hle_sce_getprio);
    REG("scePthreadSetprio", hle_sce_setprio);
    REG("_Locksyslock", hle_locksyslock);
    REG("_Unlocksyslock", hle_unlocksyslock);
    REG("_Mtx_init", hle_mtx_init);
    REG("_Mtx_destroy", hle_mtx_destroy);
    REG("_Mtx_lock", hle_mtx_lock);
    REG("_Mtx_trylock", hle_mtx_trylock);
    REG("_Mtx_unlock", hle_mtx_unlock);
    REG("_Atomic_fetch_add_4", hle_atomic_add4);
    REG("_Atomic_fetch_sub_4", hle_atomic_sub4);
#undef REG
}
