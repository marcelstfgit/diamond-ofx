#pragma once

#include <atomic>
#include <thread>

namespace diamond {

// Splits [begin, end) into bands and calls band(state, first, last) for each
// one on the system thread pool, with the calling thread taking part. Bands run
// at the same time, so a band may only write state that no other band touches.
// grain is the smallest band worth handing to another thread. Returns once all
// bands have finished and rethrows the first exception that a band threw.
void runBands(int begin, int end, int grain, void (*band)(void*, int, int), void* state);

template <typename Body>
void parallelBands(int begin, int end, const Body& body, int grain = 1) {
    runBands(begin, end, grain, [](void* state, int first, int last) {
        (*static_cast<const Body*>(state))(first, last);
    }, const_cast<void*>(static_cast<const void*>(&body)));
}

// Hosts expect their abort function to be called from the render thread. This
// wraps it so that only the thread which created the object asks the host,
// while pool threads read the latched answer.
class Cancellation {
public:
    Cancellation(bool (*abort)(void*), void* context)
        : abort_(abort), context_(context), owner_(std::this_thread::get_id()) {}

    Cancellation(const Cancellation&) = delete;
    Cancellation& operator=(const Cancellation&) = delete;

    // Matches the bool (*)(void*) abort signature used by the renderers.
    static bool poll(void* cancellation) {
        auto& self = *static_cast<Cancellation*>(cancellation);
        if (self.stopped_.load(std::memory_order_relaxed)) {
            return true;
        }
        if (!self.abort_ || std::this_thread::get_id() != self.owner_ || !self.abort_(self.context_)) {
            return false;
        }
        self.stopped_.store(true, std::memory_order_relaxed);
        return true;
    }

private:
    bool (*abort_)(void*);
    void* context_;
    std::thread::id owner_;
    std::atomic<bool> stopped_{false};
};

}  // namespace diamond
