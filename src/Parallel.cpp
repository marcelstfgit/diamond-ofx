#include "Parallel.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <execution>
#include <mutex>
#include <numeric>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define DIAMOND_HAS_MXCSR 1
#endif

namespace diamond {
namespace {

// A band that starts another parallel loop runs it inline, so pool threads
// never wait on work that only other pool threads could finish.
thread_local bool insideBand = false;

int threadCount() {
    static const int count = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return count;
}

}  // namespace

void runBands(int begin, int end, int grain, void (*band)(void*, int, int), void* state) {
    if (begin >= end) {
        return;
    }
    const std::int64_t length = static_cast<std::int64_t>(end) - begin;
    const int threads = threadCount();
    // Several bands per thread keep the load even when rows differ in cost.
    const std::int64_t count = std::min<std::int64_t>(length / std::max(1, grain), 4LL * threads);
    if (count < 2 || threads < 2 || insideBand) {
        band(state, begin, end);
        return;
    }

    std::vector<int> bands(static_cast<std::size_t>(count));
    std::iota(bands.begin(), bands.end(), 0);
    std::mutex failureLock;
    std::exception_ptr failure;
    std::atomic<bool> failed{false};
#ifdef DIAMOND_HAS_MXCSR
    // Pool threads inherit the render thread's rounding and denormal modes, so
    // a band computes the same values whichever thread happens to run it.
    const unsigned int floatControl = _mm_getcsr();
#endif

    // The parallel algorithms terminate the process when an element function
    // throws, so the first exception is carried back to the calling thread.
    std::for_each(std::execution::par, bands.begin(), bands.end(), [&](int index) {
        if (failed.load(std::memory_order_relaxed)) {
            return;
        }
        const int first = begin + static_cast<int>(length * index / count);
        const int last = begin + static_cast<int>(length * (index + 1) / count);
        const bool nested = insideBand;
        insideBand = true;
#ifdef DIAMOND_HAS_MXCSR
        const unsigned int previousControl = _mm_getcsr();
        _mm_setcsr(floatControl);
#endif
        try {
            band(state, first, last);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(failureLock);
            if (!failure) {
                failure = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
#ifdef DIAMOND_HAS_MXCSR
        _mm_setcsr(previousControl);
#endif
        insideBand = nested;
    });
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}  // namespace diamond
