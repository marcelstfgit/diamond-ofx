#include "Gpu05.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace diamond::fx {
namespace {

// Minimal OpenCL 1.2 ABI. The system ICD is optional and loaded at runtime,
// avoiding SDK/runtime dependencies in the installed Diamond bundle.
using ClInt = std::int32_t;
using ClUint = std::uint32_t;
using ClFlags = std::uint64_t;
using ClProp = std::intptr_t;
struct ClPlatformObject;
struct ClDeviceObject;
struct ClContextObject;
struct ClQueueObject;
struct ClProgramObject;
struct ClKernelObject;
struct ClMemoryObject;
struct ClEventObject;
using ClPlatform = ClPlatformObject*;
using ClDevice = ClDeviceObject*;
using ClContext = ClContextObject*;
using ClQueue = ClQueueObject*;
using ClProgram = ClProgramObject*;
using ClKernel = ClKernelObject*;
using ClMemory = ClMemoryObject*;
using ClEvent = ClEventObject*;
using ClContextCallback = void(CALLBACK*)(const char*, const void*, std::size_t, void*);
using ClBuildCallback = void(CALLBACK*)(ClProgram, void*);

constexpr ClInt kClSuccess = 0;
constexpr ClUint kClTrue = 1;
constexpr ClFlags kClDeviceGpu = 1ULL << 2;
constexpr ClFlags kClWriteOnly = 1ULL << 1;
constexpr ClFlags kClReadOnly = 1ULL << 2;
constexpr ClFlags kClCopyHostPtr = 1ULL << 5;
constexpr ClProp kClContextPlatform = 0x1084;
constexpr ClUint kClDeviceGlobalMemSize = 0x101F;
constexpr ClUint kClDeviceHostUnifiedMemory = 0x1035;

struct OpenCl {
    ClInt (WINAPI* getPlatforms)(ClUint, ClPlatform*, ClUint*) = nullptr;
    ClInt (WINAPI* getDevices)(ClPlatform, ClFlags, ClUint, ClDevice*, ClUint*) = nullptr;
    ClInt (WINAPI* getDeviceInfo)(ClDevice, ClUint, std::size_t, void*, std::size_t*) = nullptr;
    ClContext (WINAPI* createContext)(const ClProp*, ClUint, const ClDevice*, ClContextCallback,
                                     void*, ClInt*) = nullptr;
    ClInt (WINAPI* releaseContext)(ClContext) = nullptr;
    ClQueue (WINAPI* createQueue)(ClContext, ClDevice, ClFlags, ClInt*) = nullptr;
    ClInt (WINAPI* releaseQueue)(ClQueue) = nullptr;
    ClProgram (WINAPI* createProgram)(ClContext, ClUint, const char**, const std::size_t*, ClInt*) = nullptr;
    ClInt (WINAPI* buildProgram)(ClProgram, ClUint, const ClDevice*, const char*, ClBuildCallback, void*) = nullptr;
    ClInt (WINAPI* releaseProgram)(ClProgram) = nullptr;
    ClKernel (WINAPI* createKernel)(ClProgram, const char*, ClInt*) = nullptr;
    ClInt (WINAPI* releaseKernel)(ClKernel) = nullptr;
    ClInt (WINAPI* setArg)(ClKernel, ClUint, std::size_t, const void*) = nullptr;
    ClMemory (WINAPI* createBuffer)(ClContext, ClFlags, std::size_t, void*, ClInt*) = nullptr;
    ClInt (WINAPI* releaseBuffer)(ClMemory) = nullptr;
    ClInt (WINAPI* enqueue)(ClQueue, ClKernel, ClUint, const std::size_t*, const std::size_t*,
                            const std::size_t*, ClUint, const ClEvent*, ClEvent*) = nullptr;
    ClInt (WINAPI* finish)(ClQueue) = nullptr;
    ClInt (WINAPI* readBuffer)(ClQueue, ClMemory, ClUint, std::size_t, std::size_t, void*,
                               ClUint, const ClEvent*, ClEvent*) = nullptr;

    template <typename Function>
    static bool bind(HMODULE library, const char* name, Function& function) {
        const auto symbol = GetProcAddress(library, name);
        static_assert(sizeof(function) == sizeof(symbol));
        std::memcpy(&function, &symbol, sizeof(function));
        return symbol != nullptr;
    }

    bool load(HMODULE library) {
        return bind(library, "clGetPlatformIDs", getPlatforms)
            && bind(library, "clGetDeviceIDs", getDevices)
            && bind(library, "clGetDeviceInfo", getDeviceInfo)
            && bind(library, "clCreateContext", createContext)
            && bind(library, "clReleaseContext", releaseContext)
            && bind(library, "clCreateCommandQueue", createQueue)
            && bind(library, "clReleaseCommandQueue", releaseQueue)
            && bind(library, "clCreateProgramWithSource", createProgram)
            && bind(library, "clBuildProgram", buildProgram)
            && bind(library, "clReleaseProgram", releaseProgram)
            && bind(library, "clCreateKernel", createKernel)
            && bind(library, "clReleaseKernel", releaseKernel)
            && bind(library, "clSetKernelArg", setArg)
            && bind(library, "clCreateBuffer", createBuffer)
            && bind(library, "clReleaseMemObject", releaseBuffer)
            && bind(library, "clEnqueueNDRangeKernel", enqueue)
            && bind(library, "clFinish", finish)
            && bind(library, "clEnqueueReadBuffer", readBuffer);
    }
};

struct GpuImageInfo {
    int srcX, srcY, srcWidth, srcHeight;
    int outX, outY, outWidth, outHeight;
};

static_assert(sizeof(GpuImageInfo) == 32);
static_assert(sizeof(Pixel) == 4 * sizeof(float));

constexpr char kPreamble[] = R"CLC(
typedef struct {
    int srcX, srcY, srcWidth, srcHeight;
    int outX, outY, outWidth, outHeight;
} GpuImageInfo;

float4 diamond_read(__global const float4* source,GpuImageInfo image,int x,int y) {
    if (x < image.srcX || y < image.srcY || x >= image.srcX + image.srcWidth
        || y >= image.srcY + image.srcHeight) {
        return (float4)(0.0f);
    }
    return source[(size_t)(y - image.srcY) * image.srcWidth + x - image.srcX];
}

float diamond_extend(float position, int low, int size, int edge) {
    const float span = (float)(size - 1);
    const float offset = position - (float)low;
    if (edge == 2 || (offset >= 0.0f && offset <= span)) {
        return position;
    }
    if (edge == 1 || size == 1) {
        return clamp(position, (float)low, (float)(low + size - 1));
    }
    const float period = 2.0f * span;
    float phase = fmod(offset, period);
    if (phase < 0.0f) {
        phase += period;
    }
    return (float)low + fmin(phase, period - phase);
}

float4 diamond_sample(__global const float4* source,GpuImageInfo image,float x,float y,int edge) {
    x = diamond_extend(x, image.srcX, image.srcWidth, edge);
    y = diamond_extend(y, image.srcY, image.srcHeight, edge);
    if (!(x > image.srcX - 1.0f && x < image.srcX + image.srcWidth
       && y > image.srcY - 1.0f && y < image.srcY + image.srcHeight)) {
        return (float4)(0.0f);
    }
    const int x0 = convert_int_rtn(x);
    const int y0 = convert_int_rtn(y);
    const float u = x - x0;
    const float v = y - y0;
    if (x0 >= image.srcX && x0 + 1 < image.srcX + image.srcWidth
        && y0 >= image.srcY && y0 + 1 < image.srcY + image.srcHeight) {
        const size_t at = (size_t)(y0 - image.srcY) * image.srcWidth + x0 - image.srcX;
        return source[at] * ((1.0f-u)*(1.0f-v)) + source[at+1] * (u*(1.0f-v))
            + source[at+image.srcWidth] * ((1.0f-u)*v) + source[at+image.srcWidth+1] * (u*v);
    }
    return diamond_read(source,image,x0,y0) * ((1.0f-u)*(1.0f-v))
        + diamond_read(source,image,x0+1,y0) * (u*(1.0f-v))
        + diamond_read(source,image,x0,y0+1) * ((1.0f-u)*v)
        + diamond_read(source,image,x0+1,y0+1) * (u*v);
}
)CLC";

template <typename Handle>
class ReleaseHandle {
public:
    ReleaseHandle(Handle value, ClInt (WINAPI* release)(Handle)) : value_(value), release_(release) {}
    ~ReleaseHandle() { if (value_) { release_(value_); } }
    ReleaseHandle(const ReleaseHandle&) = delete;
    ReleaseHandle& operator=(const ReleaseHandle&) = delete;
    Handle get() const { return value_; }

private:
    Handle value_;
    ClInt (WINAPI* release_)(Handle);
};

class Runtime {
public:
    Runtime() {
        library_ = LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library_ || !cl_.load(library_)) {
            return;
        }
        ClUint count = 0;
        if (cl_.getPlatforms(0, nullptr, &count) != kClSuccess || count == 0) {
            return;
        }
        std::vector<ClPlatform> platforms(count);
        if (cl_.getPlatforms(count, platforms.data(), nullptr) != kClSuccess) {
            return;
        }
        struct Candidate {
            ClPlatform platform;
            ClDevice device;
            bool dedicated;
            std::uint64_t memory;
        };
        std::vector<Candidate> candidates;
        for (const auto platform : platforms) {
            std::array<ClDevice, 8> devices{};
            ClUint found = 0;
            if (cl_.getDevices(platform, kClDeviceGpu, static_cast<ClUint>(devices.size()),
                               devices.data(), &found) != kClSuccess) {
                continue;
            }
            found = std::min(found, static_cast<ClUint>(devices.size()));
            for (ClUint index = 0; index < found; ++index) {
                // A device that cannot answer is ranked with the integrated ones.
                ClUint sharesHostMemory = kClTrue;
                std::uint64_t memory = 0;
                cl_.getDeviceInfo(devices[index], kClDeviceHostUnifiedMemory, sizeof(sharesHostMemory),
                                  &sharesHostMemory, nullptr);
                cl_.getDeviceInfo(devices[index], kClDeviceGlobalMemSize, sizeof(memory), &memory, nullptr);
                candidates.push_back({platform, devices[index], sharesHostMemory == 0, memory});
            }
        }
        // Laptops report an integrated GPU next to the dedicated card, in an
        // order set by the installed drivers. Prefer a card with its own memory,
        // then the one with more of it. Integrated GPUs report shared system
        // memory, so their size only decides between two devices of one kind.
        std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.dedicated != b.dedicated ? a.dedicated : a.memory > b.memory;
        });
        for (const Candidate& candidate : candidates) {
            const ClProp properties[]{kClContextPlatform, reinterpret_cast<ClProp>(candidate.platform), 0};
            ClInt error = 0;
            const auto context = cl_.createContext(properties, 1, &candidate.device, nullptr, nullptr, &error);
            if (context && error == kClSuccess) {
                context_ = context;
                device_ = candidate.device;
                break;
            }
            if (context) {
                cl_.releaseContext(context);
            }
        }
    }

    ~Runtime() {
        for (const auto& item : programs_) {
            if (item.second) {
                cl_.releaseProgram(item.second);
            }
        }
        if (context_) {
            cl_.releaseContext(context_);
        }
        if (library_) {
            FreeLibrary(library_);
        }
    }

    ClProgram program(const char* source) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [entry, inserted] = programs_.try_emplace(source, nullptr);
        if (!inserted) {
            return entry->second;
        }
        const char* code[]{kPreamble, source};
        ClInt error = 0;
        const auto created = cl_.createProgram(context_, 2, code, nullptr, &error);
        if (created && error == kClSuccess
            && cl_.buildProgram(created, 1, &device_, "-cl-std=CL1.2", nullptr, nullptr) == kClSuccess) {
            entry->second = created;
        } else if (created) {
            cl_.releaseProgram(created);
        }
        return entry->second;
    }

    bool dispatch(const Context& context, const char* source, const char* kernelName,
                  const void* params, std::size_t paramBytes, const void* extra, std::size_t extraBytes) {
        if (!context_ || context.cancelled()) {
            return false;
        }
        const Rect src = context.source.bounds;
        const Rect dst = context.output.bounds;
        const GpuImageInfo dimensions{src.x1, src.y1, width(src), height(src),
                                      dst.x1, dst.y1, width(dst), height(dst)};
        const std::size_t inputBytes = byteCount(dimensions.srcWidth, dimensions.srcHeight);
        const std::size_t outputBytes = byteCount(dimensions.outWidth, dimensions.outHeight);
        if (inputBytes == 0 || outputBytes == 0 || !params || paramBytes == 0) {
            return false;
        }
        const auto compiled = program(source);
        if (!compiled || context.cancelled()) {
            return false;
        }

        ReleaseHandle queue(cl_.createQueue(context_, device_, 0, nullptr), cl_.releaseQueue);
        ReleaseHandle kernel(cl_.createKernel(compiled, kernelName, nullptr), cl_.releaseKernel);
        ReleaseHandle input(makeReadBuffer(inputBytes, context.source.data()), cl_.releaseBuffer);
        ReleaseHandle output(cl_.createBuffer(context_, kClWriteOnly, outputBytes, nullptr, nullptr),
                             cl_.releaseBuffer);
        ReleaseHandle param(makeReadBuffer(paramBytes, params), cl_.releaseBuffer);
        constexpr ClUint dummy = 0;
        ReleaseHandle aux(extra && extraBytes ? makeReadBuffer(extraBytes, extra)
                                             : makeReadBuffer(sizeof(dummy), &dummy), cl_.releaseBuffer);
        const ClMemory inputArg = input.get();
        const ClMemory outputArg = output.get();
        const ClMemory paramArg = param.get();
        const ClMemory extraArg = aux.get();
        if (!queue.get() || !kernel.get() || !inputArg || !outputArg || !paramArg || !extraArg
            || cl_.setArg(kernel.get(), 0, sizeof(inputArg), &inputArg) != kClSuccess
            || cl_.setArg(kernel.get(), 1, sizeof(outputArg), &outputArg) != kClSuccess
            || cl_.setArg(kernel.get(), 2, sizeof(dimensions), &dimensions) != kClSuccess
            || cl_.setArg(kernel.get(), 3, sizeof(paramArg), &paramArg) != kClSuccess
            || cl_.setArg(kernel.get(), 4, sizeof(extraArg), &extraArg) != kClSuccess
            || context.cancelled()) {
            return false;
        }
        // Keep at most one strip in flight so a cancelled host render can stop
        // between dispatches rather than waiting for the whole frame's kernel.
        constexpr std::size_t maxPixelsPerDispatch = 1u << 20;
        const auto columns = static_cast<std::size_t>(dimensions.outWidth);
        const auto rows = static_cast<std::size_t>(dimensions.outHeight);
        const auto band = std::max<std::size_t>(1, maxPixelsPerDispatch / columns);
        for (std::size_t row = 0; row < rows; row += band) {
            const std::size_t origin[]{0, row};
            const std::size_t size[]{columns, std::min(band, rows - row)};
            if (context.cancelled()
                || cl_.enqueue(queue.get(), kernel.get(), 2, origin, size, nullptr,
                               0, nullptr, nullptr) != kClSuccess) {
                return false;
            }
            if (row + band < rows && cl_.finish(queue.get()) != kClSuccess) {
                return false;
            }
        }
        return cl_.readBuffer(queue.get(), outputArg, kClTrue, 0, outputBytes, context.output.data(),
                              0, nullptr, nullptr) == kClSuccess;
    }

private:
    static std::size_t byteCount(int w, int h) {
        if (w <= 0 || h <= 0) {
            return 0;
        }
        const auto row = static_cast<std::size_t>(w) * sizeof(Pixel);
        return static_cast<std::size_t>(h) <= std::numeric_limits<std::size_t>::max() / row
            ? row * h : 0;
    }

    ClMemory makeReadBuffer(std::size_t bytes, const void* data) const {
        return cl_.createBuffer(context_, kClReadOnly | kClCopyHostPtr, bytes, const_cast<void*>(data), nullptr);
    }

    HMODULE library_ = nullptr;
    OpenCl cl_;
    ClContext context_ = nullptr;
    ClDevice device_ = nullptr;
    std::mutex mutex_;
    // Each source pointer refers to a static string literal and is unique.
    std::unordered_map<const char*, ClProgram> programs_;
};

struct RuntimeCache {
    std::mutex mutex;
    // Keep OpenCL destruction out of static object teardown under the loader lock.
    std::shared_ptr<Runtime>* instance = nullptr;
};

RuntimeCache cache;

std::shared_ptr<Runtime> acquireRuntime() {
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.instance) {
        cache.instance = new std::shared_ptr<Runtime>(std::make_shared<Runtime>());
    }
    return *cache.instance;
}

} // namespace

bool runGpu05(const Context& context, const char* programSource, const char* kernelName,
              const void* params, std::size_t paramBytes, const void* extra, std::size_t extraBytes) {
    try {
        return acquireRuntime()->dispatch(context, programSource, kernelName, params, paramBytes, extra, extraBytes);
    } catch (...) {
        return false;
    }
}

void releaseGpu05() {
    std::unique_ptr<std::shared_ptr<Runtime>> retired;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        retired.reset(cache.instance);
        cache.instance = nullptr;
    }
}

} // namespace diamond::fx
