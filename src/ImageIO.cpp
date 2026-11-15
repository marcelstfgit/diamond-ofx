#include "EffectsHost.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>

namespace diamond::fx {

void renderConfiguredEffect(Kind kind, Parameters parameters, const Context& context);

namespace {

Rect intersect(Rect a, Rect b) {
    return {std::max(a.x1, b.x1), std::max(a.y1, b.y1),
            std::min(a.x2, b.x2), std::min(a.y2, b.y2)};
}

bool empty(Rect bounds) {
    return bounds.x1 >= bounds.x2 || bounds.y1 >= bounds.y2;
}

void validateView(Rect bounds, std::ptrdiff_t rowBytes, PixelFormat format) {
    if (format.components != 1 && format.components != 3 && format.components != 4) {
        throw StatusError{kOfxStatErrFormat};
    }

    const auto columns = static_cast<std::int64_t>(bounds.x2) - bounds.x1;
    const auto rows = static_cast<std::int64_t>(bounds.y2) - bounds.y1;
    if (columns > std::numeric_limits<int>::max() || rows > std::numeric_limits<int>::max()) {
        throw StatusError{kOfxStatErrFormat};
    }

    std::uintmax_t componentBytes = 0;
    switch (format.depth) {
        case Depth::Byte: componentBytes = sizeof(std::uint8_t); break;
        case Depth::Short: componentBytes = sizeof(std::uint16_t); break;
        case Depth::Float: componentBytes = sizeof(float); break;
    }
    const auto rowSize = static_cast<std::uintmax_t>(columns) * format.components * componentBytes;
    const auto stride = rowBytes < 0
        ? static_cast<std::uintmax_t>(-(rowBytes + 1)) + 1
        : static_cast<std::uintmax_t>(rowBytes);
    const auto maxOffset = static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (componentBytes == 0 || rowSize > maxOffset ||
        (rows > 1 && (stride < rowSize || stride > (maxOffset - rowSize) / (rows - 1)))) {
        throw StatusError{kOfxStatErrFormat};
    }
}

Rect checkedSourceBounds(ImageView source, PixelFormat format) {
    if (!source.data || empty(source.bounds)) {
        return {};
    }
    validateView(source.bounds, source.rowBytes, format);
    return source.bounds;
}

template <typename Sample>
float loadComponent(const std::byte* data, int channel) {
    Sample sample;
    std::memcpy(&sample, data + channel * sizeof(Sample), sizeof(Sample));
    if constexpr (std::is_same_v<Sample, float>) {
        return sample;
    } else {
        return static_cast<float>(sample) / std::numeric_limits<Sample>::max();
    }
}

template <typename Sample>
void storeComponent(std::byte* data, int channel, float value) {
    Sample sample;
    if constexpr (std::is_same_v<Sample, float>) {
        sample = value;
    } else {
        constexpr Sample maxValue = std::numeric_limits<Sample>::max();
        if (value >= 1.0f) {
            sample = maxValue;
        } else if (value > 0.0f) {
            sample = static_cast<Sample>(value * maxValue + 0.5f);
        } else {
            sample = 0;
        }
    }
    std::memcpy(data + channel * sizeof(Sample), &sample, sizeof(Sample));
}

template <typename Sample>
Pixel readPixel(const std::byte* data, PixelFormat format) {
    if (format.components == 1) {
        const float alpha = loadComponent<Sample>(data, 0);
        return {alpha, alpha, alpha, alpha};
    }

    Pixel pixel{loadComponent<Sample>(data, 0), loadComponent<Sample>(data, 1),
                loadComponent<Sample>(data, 2), 1.0f};
    if (format.components == 4) {
        pixel.a = loadComponent<Sample>(data, 3);
        if (format.straightAlpha) {
            pixel.r *= pixel.a;
            pixel.g *= pixel.a;
            pixel.b *= pixel.a;
        }
    }
    return pixel;
}

template <typename Sample>
void writePixel(std::byte* data, PixelFormat format, Pixel pixel) {
    if (format.components == 1) {
        storeComponent<Sample>(data, 0, pixel.a);
        return;
    }
    if (format.components == 4) {
        if (format.straightAlpha) {
            const float inverseAlpha = pixel.a == 0.0f ? 0.0f : 1.0f / pixel.a;
            pixel.r *= inverseAlpha;
            pixel.g *= inverseAlpha;
            pixel.b *= inverseAlpha;
        }
        storeComponent<Sample>(data, 3, pixel.a);
    }
    storeComponent<Sample>(data, 0, pixel.r);
    storeComponent<Sample>(data, 1, pixel.g);
    storeComponent<Sample>(data, 2, pixel.b);
}

template <typename Sample>
void decode(ImageView source, PixelFormat format, Image& decoded, const Context& context) {
    const auto* data = static_cast<const std::byte*>(source.data);
    const auto pixelBytes = static_cast<std::ptrdiff_t>(sizeof(Sample)) * format.components;
    forEachRow(context, source.bounds.y1, source.bounds.y2, [&](int y) {
        const auto* pixel = data + (static_cast<std::ptrdiff_t>(y) - source.bounds.y1) * source.rowBytes;
        for (int x = source.bounds.x1; x < source.bounds.x2; ++x, pixel += pixelBytes) {
            decoded.pixel(x, y) = readPixel<Sample>(pixel, format);
        }
    });
}

template <typename Sample>
void encode(MutableImageView destination, PixelFormat format, const Context& context) {
    auto* data = static_cast<std::byte*>(destination.data);
    const auto pixelBytes = static_cast<std::ptrdiff_t>(sizeof(Sample)) * format.components;
    const Rect window = context.output.bounds;
    const auto xOffset = (static_cast<std::ptrdiff_t>(window.x1) - destination.bounds.x1) * pixelBytes;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        auto* pixel = data + (static_cast<std::ptrdiff_t>(y) - destination.bounds.y1) * destination.rowBytes
            + xOffset;
        for (int x = window.x1; x < window.x2; ++x, pixel += pixelBytes) {
            writePixel<Sample>(pixel, format, context.output.pixel(x, y));
        }
    });
}

void decodeImage(ImageView source, PixelFormat format, Image& decoded, const Context& context) {
    if (empty(decoded.bounds)) {
        return;
    }
    switch (format.depth) {
        case Depth::Byte: decode<std::uint8_t>(source, format, decoded, context); break;
        case Depth::Short: decode<std::uint16_t>(source, format, decoded, context); break;
        case Depth::Float: decode<float>(source, format, decoded, context); break;
    }
}

void encodeImage(MutableImageView destination, PixelFormat format, const Context& context) {
    if (context.cancelled()) {
        return;
    }
    switch (format.depth) {
        case Depth::Byte: encode<std::uint8_t>(destination, format, context); break;
        case Depth::Short: encode<std::uint16_t>(destination, format, context); break;
        case Depth::Float: encode<float>(destination, format, context); break;
    }
}

} // namespace

void renderEffect(Kind kind, Parameters parameters, ImageView source, PixelFormat inputFormat,
                  MutableImageView destination, PixelFormat outputFormat, Rect frame, Rect window,
                  double pixelAspect, double timeFrames, double fps, bool (*abort)(void*),
                  void* abortContext, const ShakeSettings* shake) {
    window = intersect(window, destination.bounds);
    if (!destination.data || empty(window) || (abort && abort(abortContext))) {
        return;
    }
    validateView(destination.bounds, destination.rowBytes, outputFormat);
    Image decoded(checkedSourceBounds(source, inputFormat));
    Image output(window);
    Cancellation cancellation(abort, abortContext);
    const Context context{decoded, output, frame, pixelAspect, timeFrames, fps,
                          Cancellation::poll, &cancellation};
    decodeImage(source, inputFormat, decoded, context);
    if (context.cancelled()) {
        return;
    }

    if (shake) {
        static_assert(std::is_standard_layout_v<Pixel> && sizeof(Pixel) == 4 * sizeof(float)
            && offsetof(Pixel, g) == sizeof(float) && offsetof(Pixel, b) == 2 * sizeof(float)
            && offsetof(Pixel, a) == 3 * sizeof(float));
        const auto stride = [](Rect area) {
            return static_cast<std::ptrdiff_t>(width(area)) * static_cast<std::ptrdiff_t>(sizeof(Pixel));
        };
        diamond::renderShake({decoded.data(), stride(decoded.bounds), decoded.bounds},
            {output.data(), stride(output.bounds), output.bounds}, frame, window, Depth::Float, 4,
            timeFrames / fps, 1.0 / fps, pixelAspect, false, *shake, Cancellation::poll, &cancellation);
    } else {
        renderConfiguredEffect(kind, parameters, context);
    }
    encodeImage(destination, outputFormat, context);
}

void renderTransitionEffect(TransitionKind kind, const TransitionSettings& settings, double progress,
                            ImageView from, PixelFormat fromFormat, ImageView to, PixelFormat toFormat,
                            MutableImageView destination, PixelFormat outputFormat, Rect frame, Rect window,
                            double pixelAspect, double timeFrames, double fps, bool (*abort)(void*),
                            void* abortContext) {
    window = intersect(window, destination.bounds);
    if (!destination.data || empty(window) || (abort && abort(abortContext))) {
        return;
    }
    validateView(destination.bounds, destination.rowBytes, outputFormat);
    progress = std::isfinite(progress) ? std::clamp(progress, 0.0, 1.0) : 0.0;
    Image outgoing(progress < 1.0 ? checkedSourceBounds(from, fromFormat) : Rect{});
    Image output(window);
    Cancellation cancellation(abort, abortContext);
    const Context context{outgoing, output, frame, pixelAspect, timeFrames, fps,
                          Cancellation::poll, &cancellation};
    decodeImage(from, fromFormat, outgoing, context);
    if (context.cancelled()) {
        return;
    }
    Image incoming(progress > 0.0 ? checkedSourceBounds(to, toFormat) : Rect{});
    decodeImage(to, toFormat, incoming, context);
    if (context.cancelled()) {
        return;
    }
    renderTransition(kind, context, incoming, progress, settings);
    encodeImage(destination, outputFormat, context);
}

} // namespace diamond::fx
