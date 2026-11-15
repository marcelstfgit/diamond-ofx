// Minimal OpenFX host that runs every Diamond effect on synthetic frames,
// outside a video editor. It prints one line per render with a hash of the
// output and the render time, and can compare both against an earlier run.

#include <ofxCore.h>
#include <ofxImageEffect.h>
#include <ofxParam.h>
#include <ofxProperty.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------- properties

struct Value {
    int integer = 0;
    double number = 0.0;
    void* pointer = nullptr;
    std::string text;
};

struct PropertySet {
    std::map<std::string, std::deque<Value>> entries;

    Value& slot(const char* name, int index)
    {
        auto& list = entries[name];
        const auto needed = static_cast<std::size_t>(index) + 1;
        if (list.size() < needed) {
            list.resize(needed);
        }
        return list[static_cast<std::size_t>(index)];
    }

    const Value* find(const char* name, int index = 0) const
    {
        const auto found = entries.find(name);
        if (found == entries.end() || index < 0
            || static_cast<std::size_t>(index) >= found->second.size()) {
            return nullptr;
        }
        return &found->second[static_cast<std::size_t>(index)];
    }

    int count(const char* name) const
    {
        const auto found = entries.find(name);
        return found == entries.end() ? 0 : static_cast<int>(found->second.size());
    }

    void text(const char* name, const char* value, int index = 0) { slot(name, index).text = value; }
    void integer(const char* name, int value, int index = 0) { slot(name, index).integer = value; }
    void number(const char* name, double value, int index = 0) { slot(name, index).number = value; }
    void pointer(const char* name, void* value, int index = 0) { slot(name, index).pointer = value; }
};

PropertySet* asSet(OfxPropertySetHandle handle) { return reinterpret_cast<PropertySet*>(handle); }
OfxPropertySetHandle asHandle(PropertySet* set) { return reinterpret_cast<OfxPropertySetHandle>(set); }

template <typename T, T Value::* Member>
OfxStatus setValues(OfxPropertySetHandle handle, const char* name, int first, int count, const T* values)
{
    if (!handle || !name || first < 0 || count < 0) {
        return kOfxStatErrBadHandle;
    }
    for (int i = 0; i < count; ++i) {
        asSet(handle)->slot(name, first + i).*Member = values[i];
    }
    return kOfxStatOK;
}

template <typename T, T Value::* Member>
OfxStatus getValues(OfxPropertySetHandle handle, const char* name, int first, int count, T* values)
{
    if (!handle || !name) {
        return kOfxStatErrBadHandle;
    }
    for (int i = 0; i < count; ++i) {
        const Value* value = asSet(handle)->find(name, first + i);
        if (!value) {
            return asSet(handle)->count(name) ? kOfxStatErrBadIndex : kOfxStatErrUnknown;
        }
        values[i] = value->*Member;
    }
    return kOfxStatOK;
}

OfxStatus setText(OfxPropertySetHandle handle, const char* name, int first, int count, const char* const* values)
{
    if (!handle || !name || first < 0 || count < 0) {
        return kOfxStatErrBadHandle;
    }
    for (int i = 0; i < count; ++i) {
        asSet(handle)->slot(name, first + i).text = values[i] ? values[i] : "";
    }
    return kOfxStatOK;
}

OfxStatus getText(OfxPropertySetHandle handle, const char* name, int first, int count, char** values)
{
    if (!handle || !name) {
        return kOfxStatErrBadHandle;
    }
    for (int i = 0; i < count; ++i) {
        const Value* value = asSet(handle)->find(name, first + i);
        if (!value) {
            return asSet(handle)->count(name) ? kOfxStatErrBadIndex : kOfxStatErrUnknown;
        }
        values[i] = const_cast<char*>(value->text.c_str());
    }
    return kOfxStatOK;
}

using Set = OfxPropertySetHandle;

const OfxPropertySuiteV1 propertySuite = {
    [](Set h, const char* n, int i, void* v) { return setValues<void*, &Value::pointer>(h, n, i, 1, &v); },
    [](Set h, const char* n, int i, const char* v) { return setText(h, n, i, 1, &v); },
    [](Set h, const char* n, int i, double v) { return setValues<double, &Value::number>(h, n, i, 1, &v); },
    [](Set h, const char* n, int i, int v) { return setValues<int, &Value::integer>(h, n, i, 1, &v); },
    [](Set h, const char* n, int c, void* const* v) { return setValues<void*, &Value::pointer>(h, n, 0, c, v); },
    [](Set h, const char* n, int c, const char* const* v) { return setText(h, n, 0, c, v); },
    [](Set h, const char* n, int c, const double* v) { return setValues<double, &Value::number>(h, n, 0, c, v); },
    [](Set h, const char* n, int c, const int* v) { return setValues<int, &Value::integer>(h, n, 0, c, v); },
    [](Set h, const char* n, int i, void** v) { return getValues<void*, &Value::pointer>(h, n, i, 1, v); },
    [](Set h, const char* n, int i, char** v) { return getText(h, n, i, 1, v); },
    [](Set h, const char* n, int i, double* v) { return getValues<double, &Value::number>(h, n, i, 1, v); },
    [](Set h, const char* n, int i, int* v) { return getValues<int, &Value::integer>(h, n, i, 1, v); },
    [](Set h, const char* n, int c, void** v) { return getValues<void*, &Value::pointer>(h, n, 0, c, v); },
    [](Set h, const char* n, int c, char** v) { return getText(h, n, 0, c, v); },
    [](Set h, const char* n, int c, double* v) { return getValues<double, &Value::number>(h, n, 0, c, v); },
    [](Set h, const char* n, int c, int* v) { return getValues<int, &Value::integer>(h, n, 0, c, v); },
    [](Set h, const char* n) -> OfxStatus {
        if (!h || !n) {
            return kOfxStatErrBadHandle;
        }
        asSet(h)->entries.erase(n);
        return kOfxStatOK;
    },
    [](Set h, const char* n, int* count) -> OfxStatus {
        if (!h || !n || !count) {
            return kOfxStatErrBadHandle;
        }
        *count = asSet(h)->count(n);
        return kOfxStatOK;
    },
};

// ---------------------------------------------------------------- parameters

struct Param {
    std::string name;
    std::string type;
    PropertySet properties;
    std::array<double, 4> numbers{};
    int integer = 0;
    double slope = 0.0;  // Change per frame around the render time.
};

struct ParamSet {
    PropertySet properties;
    std::map<std::string, std::unique_ptr<Param>> byName;
    std::vector<Param*> ordered;
};

constexpr double renderTime = 12.0;
constexpr double frameRate = 30.0;

// Hosts only expect calls from the thread that is running the action. The
// harness counts calls from any other thread and fails the run when it sees one.
DWORD actionThread = 0;
std::atomic<long> foreignCalls{0};
std::atomic<long> abortCalls{0};
long cancelAfter = -1;  // Report an abort after this many polls; -1 never aborts.

void noteHostCall()
{
    if (GetCurrentThreadId() != actionThread) {
        ++foreignCalls;
    }
}

bool isType(const Param& param, const char* type) { return param.type == type; }

OfxStatus readParam(const Param& param, double time, va_list arguments)
{
    noteHostCall();
    if (isType(param, kOfxParamTypeDouble)) {
        *va_arg(arguments, double*) = param.numbers[0] + param.slope * (time - renderTime);
    } else if (isType(param, kOfxParamTypeInteger) || isType(param, kOfxParamTypeChoice)
               || isType(param, kOfxParamTypeBoolean)) {
        *va_arg(arguments, int*) = param.integer;
    } else if (isType(param, kOfxParamTypeRGB) || isType(param, kOfxParamTypeRGBA)
               || isType(param, kOfxParamTypeDouble2D)) {
        const int count = isType(param, kOfxParamTypeRGBA) ? 4 : isType(param, kOfxParamTypeRGB) ? 3 : 2;
        for (int i = 0; i < count; ++i) {
            *va_arg(arguments, double*) = param.numbers[static_cast<std::size_t>(i)];
        }
    } else {
        return kOfxStatErrUnsupported;
    }
    return kOfxStatOK;
}

OfxStatus paramDefine(OfxParamSetHandle handle, const char* type, const char* name, OfxPropertySetHandle* properties)
{
    auto* set = reinterpret_cast<ParamSet*>(handle);
    if (!set || !type || !name) {
        return kOfxStatErrBadHandle;
    }
    if (set->byName.count(name)) {
        return kOfxStatErrExists;
    }
    auto param = std::make_unique<Param>();
    param->name = name;
    param->type = type;
    if (properties) {
        *properties = asHandle(&param->properties);
    }
    set->ordered.push_back(param.get());
    set->byName[name] = std::move(param);
    return kOfxStatOK;
}

OfxStatus paramGetHandle(OfxParamSetHandle handle, const char* name, OfxParamHandle* param,
                         OfxPropertySetHandle* properties)
{
    auto* set = reinterpret_cast<ParamSet*>(handle);
    if (!set || !name) {
        return kOfxStatErrBadHandle;
    }
    const auto found = set->byName.find(name);
    if (found == set->byName.end()) {
        return kOfxStatErrUnknown;
    }
    if (param) {
        *param = reinterpret_cast<OfxParamHandle>(found->second.get());
    }
    if (properties) {
        *properties = asHandle(&found->second->properties);
    }
    return kOfxStatOK;
}

OfxStatus paramSetGetPropertySet(OfxParamSetHandle handle, OfxPropertySetHandle* properties)
{
    *properties = asHandle(&reinterpret_cast<ParamSet*>(handle)->properties);
    return kOfxStatOK;
}

OfxStatus paramGetPropertySet(OfxParamHandle handle, OfxPropertySetHandle* properties)
{
    *properties = asHandle(&reinterpret_cast<Param*>(handle)->properties);
    return kOfxStatOK;
}

OfxStatus paramGetValue(OfxParamHandle handle, ...)
{
    va_list arguments;
    va_start(arguments, handle);
    const auto status = readParam(*reinterpret_cast<Param*>(handle), renderTime, arguments);
    va_end(arguments);
    return status;
}

OfxStatus paramGetValueAtTime(OfxParamHandle handle, OfxTime time, ...)
{
    va_list arguments;
    va_start(arguments, time);
    const auto status = readParam(*reinterpret_cast<Param*>(handle), time, arguments);
    va_end(arguments);
    return status;
}

OfxStatus paramGetDerivative(OfxParamHandle, OfxTime, ...) { return kOfxStatErrUnsupported; }
OfxStatus paramGetIntegral(OfxParamHandle, OfxTime, OfxTime, ...) { return kOfxStatErrUnsupported; }
OfxStatus paramSetValue(OfxParamHandle, ...) { return kOfxStatErrUnsupported; }
OfxStatus paramSetValueAtTime(OfxParamHandle, OfxTime, ...) { return kOfxStatErrUnsupported; }

const OfxParameterSuiteV1 parameterSuite = {
    paramDefine,
    paramGetHandle,
    paramSetGetPropertySet,
    paramGetPropertySet,
    paramGetValue,
    paramGetValueAtTime,
    paramGetDerivative,
    paramGetIntegral,
    paramSetValue,
    paramSetValueAtTime,
    [](OfxParamHandle, unsigned int* keys) -> OfxStatus { *keys = 0; return kOfxStatOK; },
    [](OfxParamHandle, unsigned int, OfxTime*) -> OfxStatus { return kOfxStatErrBadIndex; },
    [](OfxParamHandle, OfxTime, int, int*) -> OfxStatus { return kOfxStatFailed; },
    [](OfxParamHandle, OfxTime) -> OfxStatus { return kOfxStatErrUnsupported; },
    [](OfxParamHandle) -> OfxStatus { return kOfxStatErrUnsupported; },
    [](OfxParamHandle, OfxParamHandle, OfxTime, const OfxRangeD*) -> OfxStatus { return kOfxStatErrUnsupported; },
    [](OfxParamSetHandle, const char*) -> OfxStatus { return kOfxStatOK; },
    [](OfxParamSetHandle) -> OfxStatus { return kOfxStatOK; },
};

// --------------------------------------------------------------------- clips

struct Depth {
    const char* name;
    const char* ofx;
    int bytes;
};

struct Layout {
    const char* name;
    const char* components;
    const char* premultiplication;
    int channels;
    bool straight;
};

constexpr Depth depths[] = {
    {"byte", kOfxBitDepthByte, 1},
    {"short", kOfxBitDepthShort, 2},
    {"float", kOfxBitDepthFloat, 4},
};

constexpr Layout layouts[] = {
    {"rgba-premult", kOfxImageComponentRGBA, kOfxImagePreMultiplied, 4, false},
    {"rgba-straight", kOfxImageComponentRGBA, kOfxImageUnPreMultiplied, 4, true},
    {"rgb", kOfxImageComponentRGB, kOfxImageOpaque, 3, false},
};

struct Frame {
    int width = 0;
    int height = 0;
    const Depth* depth = nullptr;
    const Layout* layout = nullptr;
    std::vector<unsigned char> bytes;

    int rowBytes() const { return width * layout->channels * depth->bytes; }

    void allocate(int w, int h, const Depth& d, const Layout& l)
    {
        width = w;
        height = h;
        depth = &d;
        layout = &l;
        bytes.assign(static_cast<std::size_t>(rowBytes()) * static_cast<std::size_t>(h), 0);
    }
};

struct Clip {
    std::string name;
    PropertySet properties;
    Frame* frame = nullptr;
};

struct Effect {
    PropertySet properties;
    ParamSet params;
    std::map<std::string, std::unique_ptr<Clip>> clips;
};

Effect* asEffect(OfxImageEffectHandle handle) { return reinterpret_cast<Effect*>(handle); }

OfxStatus clipDefine(OfxImageEffectHandle handle, const char* name, OfxPropertySetHandle* properties)
{
    auto& clip = asEffect(handle)->clips[name];
    if (!clip) {
        clip = std::make_unique<Clip>();
        clip->name = name;
    }
    if (properties) {
        *properties = asHandle(&clip->properties);
    }
    return kOfxStatOK;
}

OfxStatus clipGetHandle(OfxImageEffectHandle handle, const char* name, OfxImageClipHandle* clip,
                        OfxPropertySetHandle* properties)
{
    const auto found = asEffect(handle)->clips.find(name);
    if (found == asEffect(handle)->clips.end()) {
        return kOfxStatErrBadHandle;
    }
    if (clip) {
        *clip = reinterpret_cast<OfxImageClipHandle>(found->second.get());
    }
    if (properties) {
        *properties = asHandle(&found->second->properties);
    }
    return kOfxStatOK;
}

OfxStatus clipGetImage(OfxImageClipHandle handle, OfxTime, const OfxRectD*, OfxPropertySetHandle* image)
{
    auto* clip = reinterpret_cast<Clip*>(handle);
    if (!clip || !clip->frame) {
        return kOfxStatFailed;
    }
    const Frame& frame = *clip->frame;
    auto* set = new PropertySet;
    set->pointer(kOfxImagePropData, const_cast<unsigned char*>(frame.bytes.data()));
    set->integer(kOfxImagePropRowBytes, frame.rowBytes());
    const int bounds[] = {0, 0, frame.width, frame.height};
    for (int i = 0; i < 4; ++i) {
        set->integer(kOfxImagePropBounds, bounds[i], i);
        set->integer(kOfxImagePropRegionOfDefinition, bounds[i], i);
    }
    set->text(kOfxImageEffectPropPixelDepth, frame.depth->ofx);
    set->text(kOfxImageEffectPropComponents, frame.layout->components);
    set->text(kOfxImageEffectPropPreMultiplication, frame.layout->premultiplication);
    set->text(kOfxImagePropField, kOfxImageFieldNone);
    set->number(kOfxImagePropPixelAspectRatio, 1.0);
    set->number(kOfxImageEffectPropRenderScale, 1.0, 0);
    set->number(kOfxImageEffectPropRenderScale, 1.0, 1);
    *image = asHandle(set);
    return kOfxStatOK;
}

OfxStatus clipGetRegionOfDefinition(OfxImageClipHandle handle, OfxTime, OfxRectD* bounds)
{
    auto* clip = reinterpret_cast<Clip*>(handle);
    if (!clip || !clip->frame) {
        return kOfxStatFailed;
    }
    *bounds = {0.0, 0.0, static_cast<double>(clip->frame->width), static_cast<double>(clip->frame->height)};
    return kOfxStatOK;
}

const OfxImageEffectSuiteV1 imageEffectSuite = {
    [](OfxImageEffectHandle h, OfxPropertySetHandle* p) -> OfxStatus {
        *p = asHandle(&asEffect(h)->properties);
        return kOfxStatOK;
    },
    [](OfxImageEffectHandle h, OfxParamSetHandle* p) -> OfxStatus {
        *p = reinterpret_cast<OfxParamSetHandle>(&asEffect(h)->params);
        return kOfxStatOK;
    },
    clipDefine,
    clipGetHandle,
    [](OfxImageClipHandle h, OfxPropertySetHandle* p) -> OfxStatus {
        *p = asHandle(&reinterpret_cast<Clip*>(h)->properties);
        return kOfxStatOK;
    },
    clipGetImage,
    [](OfxPropertySetHandle image) -> OfxStatus {
        delete asSet(image);
        return kOfxStatOK;
    },
    clipGetRegionOfDefinition,
    [](OfxImageEffectHandle) -> int {
        noteHostCall();
        const long polls = ++abortCalls;
        return cancelAfter >= 0 && polls > cancelAfter ? 1 : 0;
    },
    [](OfxImageEffectHandle, size_t, OfxImageMemoryHandle*) -> OfxStatus { return kOfxStatErrUnsupported; },
    [](OfxImageMemoryHandle) -> OfxStatus { return kOfxStatErrUnsupported; },
    [](OfxImageMemoryHandle, void**) -> OfxStatus { return kOfxStatErrUnsupported; },
    [](OfxImageMemoryHandle) -> OfxStatus { return kOfxStatErrUnsupported; },
};

const void* fetchSuite(OfxPropertySetHandle, const char* name, int version)
{
    if (version != 1) {
        return nullptr;
    }
    if (std::strcmp(name, kOfxPropertySuite) == 0) {
        return &propertySuite;
    }
    if (std::strcmp(name, kOfxParameterSuite) == 0) {
        return &parameterSuite;
    }
    if (std::strcmp(name, kOfxImageEffectSuite) == 0) {
        return &imageEffectSuite;
    }
    return nullptr;
}

// ------------------------------------------------------------- test pictures

std::uint32_t mixBits(std::uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

// A gradient with fine noise, a few highlights brighter than white and a soft
// transparent border, so thresholds, blurs and alpha handling all have work.
std::array<float, 4> picture(int x, int y, int width, int height, int variant)
{
    const double u = (x + 0.5) / width;
    const double v = (y + 0.5) / height;
    const double turn = variant == 0 ? 1.0 : -1.0;
    double red = 0.45 + 0.35 * std::sin(9.0 * u * turn + 3.0 * v);
    double green = 0.15 + 0.7 * u * v + 0.1 * std::cos(14.0 * v);
    double blue = variant == 0 ? 0.85 - 0.7 * u : 0.2 + 0.6 * v;
    const double spots[3][2] = {{0.22, 0.70}, {0.68, 0.34}, {0.86, 0.82}};
    for (int i = 0; i < 3; ++i) {
        const double dx = (u - spots[i][variant]) * width / height;
        const double dy = v - spots[i][1 - variant];
        const double glow = 1.6 * std::exp(-(dx * dx + dy * dy) / 0.0009);
        red += glow;
        green += glow * 0.9;
        blue += glow * 0.7;
    }
    const auto grain = mixBits(static_cast<std::uint32_t>(x) * 0x9e3779b9u
        ^ static_cast<std::uint32_t>(y) * 0x85ebca6bu ^ static_cast<std::uint32_t>(variant));
    const double noise = (static_cast<double>(grain >> 8) / 16777216.0 - 0.5) * 0.08;
    const double edge = std::min({u, v, 1.0 - u, 1.0 - v});
    const double alpha = std::clamp(edge / 0.06, 0.0, 1.0);
    return {static_cast<float>(std::max(0.0, red + noise)), static_cast<float>(std::max(0.0, green + noise)),
            static_cast<float>(std::max(0.0, blue + noise)), static_cast<float>(alpha)};
}

void storeSample(unsigned char* destination, const Depth& depth, float value)
{
    if (depth.bytes == 4) {
        std::memcpy(destination, &value, sizeof(value));
    } else if (depth.bytes == 2) {
        const auto sample = static_cast<std::uint16_t>(std::clamp(value, 0.0f, 1.0f) * 65535.0f + 0.5f);
        std::memcpy(destination, &sample, sizeof(sample));
    } else {
        *destination = static_cast<unsigned char>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
}

void paint(Frame& frame, int variant)
{
    const int channels = frame.layout->channels;
    const int sampleBytes = frame.depth->bytes;
    for (int y = 0; y < frame.height; ++y) {
        auto* row = frame.bytes.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.rowBytes());
        for (int x = 0; x < frame.width; ++x) {
            auto pixel = picture(x, y, frame.width, frame.height, variant);
            if (channels == 4 && !frame.layout->straight) {
                for (int c = 0; c < 3; ++c) {
                    pixel[static_cast<std::size_t>(c)] *= pixel[3];
                }
            }
            for (int c = 0; c < channels; ++c) {
                storeSample(row + (x * channels + c) * sampleBytes, *frame.depth, pixel[static_cast<std::size_t>(c)]);
            }
        }
    }
}

std::string hashOf(const std::vector<unsigned char>& bytes)
{
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const unsigned char byte : bytes) {
        hash = (hash ^ byte) * 0x100000001b3ull;
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

void writePicture(const Frame& frame, const std::string& path)
{
    if (frame.depth->bytes != 1) {
        return;
    }
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << frame.width << ' ' << frame.height << "\n255\n";
    const int channels = frame.layout->channels;
    for (int y = frame.height - 1; y >= 0; --y) {
        const auto* row = frame.bytes.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(frame.rowBytes());
        for (int x = 0; x < frame.width; ++x) {
            file.write(reinterpret_cast<const char*>(row + x * channels), 3);
        }
    }
}

// ---------------------------------------------------------------- parameters

double defaultNumber(const Param& param, const char* name, int index, double fallback)
{
    const Value* value = param.properties.find(name, index);
    return value ? value->number : fallback;
}

int defaultInteger(const Param& param, const char* name, int fallback)
{
    const Value* value = param.properties.find(name);
    return value ? value->integer : fallback;
}

// "defaults" uses the values each effect declares. "varied" moves every control
// away from its default and animates the motion-blur transform, so code paths
// that the defaults leave idle are rendered too.
void configure(ParamSet& params, bool varied)
{
    for (Param* param : params.ordered) {
        param->slope = 0.0;
        if (param->name == kOfxImageEffectTransitionParamName) {
            param->numbers[0] = varied ? 0.3 : 0.5;
        } else if (isType(*param, kOfxParamTypeDouble)) {
            const double initial = defaultNumber(*param, kOfxParamPropDefault, 0, 0.0);
            const double low = defaultNumber(*param, kOfxParamPropMin, 0, initial);
            const double high = defaultNumber(*param, kOfxParamPropMax, 0, initial);
            param->numbers[0] = varied && high - low < 1000.0 ? low + 0.2 * (high - low) : initial;
            if (varied && (param->name == "shiftX" || param->name == "rotation")) {
                param->numbers[0] = initial;
                param->slope = param->name == "shiftX" ? 3.0 : 2.0;
            }
        } else if (isType(*param, kOfxParamTypeInteger)) {
            const int initial = defaultInteger(*param, kOfxParamPropDefault, 0);
            const int low = defaultInteger(*param, kOfxParamPropMin, initial);
            const int high = defaultInteger(*param, kOfxParamPropMax, initial);
            param->integer = !varied ? initial
                : static_cast<std::int64_t>(high) - low <= 1000 ? low + (high - low) * 35 / 100 : initial + 7;
        } else if (isType(*param, kOfxParamTypeChoice)) {
            const int initial = defaultInteger(*param, kOfxParamPropDefault, 0);
            const int options = std::max(1, param->properties.count(kOfxParamPropChoiceOption));
            param->integer = varied ? (initial + 1) % options : initial;
        } else if (isType(*param, kOfxParamTypeBoolean)) {
            const int initial = defaultInteger(*param, kOfxParamPropDefault, 0);
            param->integer = varied ? !initial : initial;
        } else if (isType(*param, kOfxParamTypeRGB)) {
            for (int i = 0; i < 3; ++i) {
                const double initial = defaultNumber(*param, kOfxParamPropDefault, i, 1.0);
                param->numbers[static_cast<std::size_t>(i)] = varied ? 0.15 + 0.7 * initial : initial;
            }
        }
    }
}

// ------------------------------------------------------------------- running

struct Options {
    std::string plugin;
    int width = 640;
    int height = 360;
    int repeat = 1;
    std::string depthFilter = "byte,short,float";
    std::string layoutFilter = "rgba-premult,rgba-straight,rgb";
    std::string only;
    std::string writePath;
    std::string comparePath;
    std::string picturePath;
};

struct Result {
    std::string effect;
    std::string scenario;
    std::string hash;
    double milliseconds = 0.0;
};

bool listed(const std::string& list, const char* name)
{
    std::stringstream stream(list);
    for (std::string item; std::getline(stream, item, ',');) {
        if (item == name) {
            return true;
        }
    }
    return false;
}

void describeFormat(PropertySet& properties, const Depth& depth, const Layout& layout)
{
    properties.text(kOfxImageEffectPropPixelDepth, depth.ofx);
    properties.text(kOfxImageEffectPropComponents, layout.components);
    properties.text(kOfxImageEffectPropPreMultiplication, layout.premultiplication);
    properties.number(kOfxImageEffectPropFrameRate, frameRate);
    properties.number(kOfxImagePropPixelAspectRatio, 1.0);
    properties.integer(kOfxImageClipPropConnected, 1);
}

void renderArguments(PropertySet& arguments, int width, int height)
{
    arguments.number(kOfxPropTime, renderTime);
    arguments.text(kOfxImageEffectPropFieldToRender, kOfxImageFieldNone);
    const int window[] = {0, 0, width, height};
    for (int i = 0; i < 4; ++i) {
        arguments.integer(kOfxImageEffectPropRenderWindow, window[i], i);
    }
    arguments.number(kOfxImageEffectPropRenderScale, 1.0, 0);
    arguments.number(kOfxImageEffectPropRenderScale, 1.0, 1);
}

bool runEffect(OfxPlugin& plugin, const Options& options, std::vector<Result>& results)
{
    Effect effect;
    const auto handle = reinterpret_cast<const void*>(&effect);
    if (plugin.mainEntry(kOfxActionDescribe, handle, nullptr, nullptr) != kOfxStatOK) {
        std::printf("%s: describe failed\n", plugin.pluginIdentifier);
        return false;
    }
    const Value* label = effect.properties.find(kOfxPropLabel);
    const std::string name = label ? label->text : plugin.pluginIdentifier;
    if (!options.only.empty() && name.find(options.only) == std::string::npos) {
        return true;
    }
    const Value* context = effect.properties.find(kOfxImageEffectPropSupportedContexts);
    if (!context) {
        std::printf("%s: no supported context\n", name.c_str());
        return false;
    }
    const bool transition = context->text == kOfxImageEffectContextTransition;
    effect.properties.text(kOfxImageEffectPropContext, context->text.c_str());
    effect.properties.number(kOfxImageEffectPropFrameRate, frameRate);
    PropertySet contextArguments;
    contextArguments.text(kOfxImageEffectPropContext, context->text.c_str());
    if (plugin.mainEntry(kOfxImageEffectActionDescribeInContext, handle, asHandle(&contextArguments), nullptr) != kOfxStatOK
        || plugin.mainEntry(kOfxActionCreateInstance, handle, nullptr, nullptr) != kOfxStatOK) {
        std::printf("%s: describe in context failed\n", name.c_str());
        return false;
    }

    bool succeeded = true;
    for (const Depth& depth : depths) {
        if (!listed(options.depthFilter, depth.name)) {
            continue;
        }
        for (const Layout& layout : layouts) {
            if (!listed(options.layoutFilter, layout.name)) {
                continue;
            }
            Frame first, second, output;
            first.allocate(options.width, options.height, depth, layout);
            paint(first, 0);
            if (transition) {
                second.allocate(options.width, options.height, depth, layout);
                paint(second, 1);
            }
            for (auto& entry : effect.clips) {
                describeFormat(entry.second->properties, depth, layout);
                entry.second->frame = entry.first == kOfxImageEffectOutputClipName ? &output
                    : entry.first == kOfxImageEffectTransitionSourceToClipName ? &second : &first;
            }

            for (const bool varied : {false, true}) {
                configure(effect.params, varied);
                Result result{name, std::string(depth.name) + '/' + layout.name + '/' + (varied ? "varied" : "defaults"),
                              "", 0.0};
                PropertySet arguments, reply;
                renderArguments(arguments, options.width, options.height);
                plugin.mainEntry(kOfxImageEffectActionGetClipPreferences, handle, nullptr, asHandle(&reply));
                plugin.mainEntry(kOfxImageEffectActionGetRegionsOfInterest, handle, asHandle(&arguments), asHandle(&reply));
                if (plugin.mainEntry(kOfxImageEffectActionIsIdentity, handle, asHandle(&arguments), asHandle(&reply))
                    == kOfxStatOK) {
                    result.hash = "identity";
                    results.push_back(result);
                    continue;
                }

                double best = 0.0;
                OfxStatus status = kOfxStatOK;
                for (int run = 0; run < options.repeat && status == kOfxStatOK; ++run) {
                    output.allocate(options.width, options.height, depth, layout);
                    abortCalls = 0;
                    const auto start = std::chrono::steady_clock::now();
                    status = plugin.mainEntry(kOfxImageEffectActionRender, handle, asHandle(&arguments), nullptr);
                    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
                    best = run == 0 ? elapsed.count() : std::min(best, elapsed.count());
                }
                if (status != kOfxStatOK) {
                    std::printf("%s %s: render returned %d\n", name.c_str(), result.scenario.c_str(), status);
                    result.hash = "failed";
                    succeeded = false;
                } else {
                    // A cancelled render leaves a partial frame that no host would use.
                    result.hash = cancelAfter >= 0 ? "cancelled" : hashOf(output.bytes);
                }
                result.milliseconds = best;
                results.push_back(result);
                if (!options.picturePath.empty() && layout.channels >= 3) {
                    writePicture(output, options.picturePath + '\\' + name + (varied ? "-varied.ppm" : "-defaults.ppm"));
                }
            }
        }
    }
    plugin.mainEntry(kOfxActionDestroyInstance, handle, nullptr, nullptr);
    return succeeded;
}

std::string key(const Result& result) { return result.effect + '\t' + result.scenario; }

int report(const std::vector<Result>& results, const Options& options)
{
    std::map<std::string, Result> earlier;
    if (!options.comparePath.empty()) {
        std::ifstream file(options.comparePath);
        for (std::string line; std::getline(file, line);) {
            std::stringstream stream(line);
            Result result;
            std::string time;
            if (std::getline(stream, result.effect, '\t') && std::getline(stream, result.scenario, '\t')
                && std::getline(stream, result.hash, '\t') && std::getline(stream, time)) {
                result.milliseconds = std::atof(time.c_str());
                earlier[key(result)] = result;
            }
        }
        if (earlier.empty()) {
            std::printf("No results could be read from %s\n", options.comparePath.c_str());
            return 1;
        }
    }

    int different = 0;
    int missing = 0;
    double total = 0.0;
    double earlierTotal = 0.0;
    for (const Result& result : results) {
        total += result.milliseconds;
        if (earlier.empty()) {
            std::printf("%-18s %-30s %-16s %9.1f ms\n", result.effect.c_str(), result.scenario.c_str(),
                        result.hash.c_str(), result.milliseconds);
            continue;
        }
        const auto found = earlier.find(key(result));
        if (found == earlier.end()) {
            ++missing;
            std::printf("%-18s %-30s NEW\n", result.effect.c_str(), result.scenario.c_str());
            continue;
        }
        const bool same = found->second.hash == result.hash;
        different += same ? 0 : 1;
        earlierTotal += found->second.milliseconds;
        const double ratio = result.milliseconds > 0.0 ? found->second.milliseconds / result.milliseconds : 0.0;
        std::printf("%-18s %-30s %-9s %9.1f -> %8.1f ms  %5.2fx\n", result.effect.c_str(), result.scenario.c_str(),
                    same ? "same" : "DIFFERENT", found->second.milliseconds, result.milliseconds, ratio);
    }
    if (earlier.empty()) {
        std::printf("%zu renders, %.0f ms in total\n", results.size(), total);
    } else {
        std::printf("%zu renders: %d different, %d without an earlier result. Total %.0f -> %.0f ms (%.2fx)\n",
                    results.size(), different, missing, earlierTotal, total, total > 0.0 ? earlierTotal / total : 0.0);
    }

    if (!options.writePath.empty()) {
        std::ofstream file(options.writePath);
        for (const Result& result : results) {
            file << result.effect << '\t' << result.scenario << '\t' << result.hash << '\t' << result.milliseconds << '\n';
        }
    }
    return different == 0 ? 0 : 1;
}

bool parse(int count, char** arguments, Options& options)
{
    for (int i = 1; i < count; ++i) {
        const std::string argument = arguments[i];
        const auto next = [&]() -> std::string { return i + 1 < count ? arguments[++i] : ""; };
        if (argument == "--size") {
            const std::string size = next();
            const auto separator = size.find('x');
            if (separator == std::string::npos) {
                return false;
            }
            options.width = std::atoi(size.substr(0, separator).c_str());
            options.height = std::atoi(size.substr(separator + 1).c_str());
        } else if (argument == "--repeat") {
            options.repeat = std::max(1, std::atoi(next().c_str()));
        } else if (argument == "--depths") {
            options.depthFilter = next();
        } else if (argument == "--layouts") {
            options.layoutFilter = next();
        } else if (argument == "--only") {
            options.only = next();
        } else if (argument == "--write") {
            options.writePath = next();
        } else if (argument == "--compare") {
            options.comparePath = next();
        } else if (argument == "--pictures") {
            options.picturePath = next();
        } else if (argument == "--cancel-after") {
            cancelAfter = std::max(0, std::atoi(next().c_str()));
        } else if (options.plugin.empty() && argument.rfind("--", 0) != 0) {
            options.plugin = argument;
        } else {
            return false;
        }
    }
    return !options.plugin.empty() && options.width > 0 && options.height > 0;
}

} // namespace

int main(int count, char** arguments)
{
    Options options;
    if (!parse(count, arguments, options)) {
        std::printf("Usage: Harness <Diamond.ofx> [--size 640x360] [--repeat 1] [--depths byte,short,float]\n"
                    "               [--layouts rgba-premult,rgba-straight,rgb] [--only D_Glow]\n"
                    "               [--write results.tsv] [--compare results.tsv] [--pictures folder]\n"
                    "               [--cancel-after polls]\n");
        return 2;
    }
    actionThread = GetCurrentThreadId();

    const HMODULE library = LoadLibraryA(options.plugin.c_str());
    if (!library) {
        std::printf("Could not load %s (error %lu)\n", options.plugin.c_str(), GetLastError());
        return 2;
    }
    using CountFunction = int (*)();
    using PluginFunction = OfxPlugin* (*)(int);
    const auto pluginCount = reinterpret_cast<CountFunction>(
        reinterpret_cast<void*>(GetProcAddress(library, "OfxGetNumberOfPlugins")));
    const auto pluginAt = reinterpret_cast<PluginFunction>(
        reinterpret_cast<void*>(GetProcAddress(library, "OfxGetPlugin")));
    if (!pluginCount || !pluginAt) {
        std::printf("%s does not export the OpenFX entry points\n", options.plugin.c_str());
        return 2;
    }

    PropertySet hostProperties;
    hostProperties.text(kOfxPropName, "diamond.test.harness");
    hostProperties.text(kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA, 0);
    hostProperties.text(kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGB, 1);
    hostProperties.text(kOfxImageEffectPropSupportedComponents, kOfxImageComponentAlpha, 2);
    OfxHost host{asHandle(&hostProperties), fetchSuite};

    std::vector<Result> results;
    std::vector<OfxPlugin*> loaded;
    bool succeeded = true;
    for (int index = 0; index < pluginCount(); ++index) {
        OfxPlugin* plugin = pluginAt(index);
        if (!plugin || std::strcmp(plugin->pluginApi, kOfxImageEffectPluginApi) != 0) {
            continue;
        }
        plugin->setHost(&host);
        if (plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) != kOfxStatOK) {
            std::printf("%s: load failed\n", plugin->pluginIdentifier);
            succeeded = false;
            continue;
        }
        loaded.push_back(plugin);
        succeeded = runEffect(*plugin, options, results) && succeeded;
    }
    for (OfxPlugin* plugin : loaded) {
        plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
    }
    FreeLibrary(library);

    const int status = report(results, options);
    if (foreignCalls > 0) {
        std::printf("The plug-in called the host %ld times from a thread other than the render thread\n",
                    foreignCalls.load());
        return 1;
    }
    return succeeded ? status : 1;
}
