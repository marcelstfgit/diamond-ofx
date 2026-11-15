#include "Shake.h"

#include "Parallel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace diamond {
namespace {

constexpr int kMaxSamples = 16;
constexpr double kRadiansPerDegree = 0.01745329251994329577;
constexpr double kNoisePeriod = 4294967296.0;

bool isEmpty(Rect rect) {
    return rect.x1 >= rect.x2 || rect.y1 >= rect.y2;
}

Rect intersect(Rect a, Rect b) {
    return {
        std::max(a.x1, b.x1), std::max(a.y1, b.y1),
        std::min(a.x2, b.x2), std::min(a.y2, b.y2)
    };
}

// Integer hashing gives each axis and octave an independent, repeatable signal.
double latticeValue(std::uint32_t index, std::uint32_t key) {
    std::uint32_t bits = index ^ key;
    bits ^= bits >> 16;
    bits *= 0x7feb352dU;
    bits ^= bits >> 15;
    bits *= 0x846ca68bU;
    bits ^= bits >> 16;
    return (static_cast<double>(bits) / 4294967295.0) * 2.0 - 1.0;
}

double smoothNoise(double time, std::uint32_t key) {
    if (!std::isfinite(time)) {
        return 0.0;
    }
    const double cell = std::floor(time);
    double wrappedCell = std::fmod(cell, kNoisePeriod);
    if (wrappedCell < 0.0) {
        wrappedCell += kNoisePeriod;
    }
    const auto index = static_cast<std::uint32_t>(wrappedCell);
    const double fraction = time - cell;
    const double blend = fraction * fraction * fraction
        * (fraction * (fraction * 6.0 - 15.0) + 10.0);
    const double left = latticeValue(index, key);
    const double right = latticeValue(index + 1U, key);
    return left + (right - left) * blend;
}

double handheldNoise(double time, std::uint32_t key) {
    return 0.62 * smoothNoise(time, key)
        + 0.28 * smoothNoise(time * 2.31 + 23.7, key ^ 0x3c6ef372U)
        + 0.10 * smoothNoise(time * 5.17 + 71.3, key ^ 0xa54ff53aU);
}

struct Transform {
    double xx, xy, xOffset;
    double yx, yy, yOffset;
};

Transform makeTransform(Rect frame, double seconds, double pixelAspect,
                        const ShakeSettings& settings) {
    const double phase = seconds * settings.frequency;
    const auto seed = static_cast<std::uint32_t>(settings.seed) * 0x9e3779b9U;
    const double width = static_cast<double>(frame.x2) - frame.x1;
    const double height = static_cast<double>(frame.y2) - frame.y1;
    const double centerX = frame.x1 + (width - 1.0) * 0.5;
    const double centerY = frame.y1 + (height - 1.0) * 0.5;

    const double shiftX = width * 0.01 * settings.xPercent * settings.amplitude
        * handheldNoise(phase + 13.1, seed ^ 0x510e527fU);
    const double shiftY = height * 0.01 * settings.yPercent * settings.amplitude
        * handheldNoise(phase + 37.9, seed ^ 0x1f83d9abU);
    const double angle = kRadiansPerDegree * settings.tiltDegrees * settings.amplitude
        * handheldNoise(phase + 7.3, seed ^ 0x5be0cd19U);
    const double zoom = 0.01 * settings.zoomPercent * settings.amplitude
        * handheldNoise(phase + 67.4, seed ^ 0x6a09e667U);
    const double scale = (1.0 + settings.overscanPercent * 0.01) * std::exp(zoom);
    const double cosine = std::cos(angle) / scale;
    const double sine = std::sin(angle) / scale;

    // Map output pixel centers back into the source. Rotation happens in
    // display space so it remains correct for non-square pixels.
    const double xy = sine / pixelAspect;
    const double yx = -sine * pixelAspect;
    const double originX = centerX + shiftX;
    const double originY = centerY + shiftY;
    return {
        cosine, xy, centerX - cosine * originX - xy * originY,
        yx, cosine, centerY - yx * originX - cosine * originY
    };
}

double reflectCoordinate(double coordinate, int begin, int end) {
    const double span = static_cast<double>(end) - begin - 1.0;
    if (span == 0.0) {
        return begin;
    }
    const double period = 2.0 * span;
    double offset = std::fmod(coordinate - begin, period);
    if (offset < 0.0) {
        offset += period;
    }
    return begin + (offset <= span ? offset : period - offset);
}

double edgeCoordinate(double coordinate, int begin, int end, Edge edge) {
    const double last = static_cast<double>(end) - 1.0;
    if (edge == Edge::Transparent || (coordinate >= begin && coordinate <= last)) {
        return coordinate;
    }
    if (edge == Edge::Clamp) {
        return std::clamp(coordinate, static_cast<double>(begin), last);
    }
    return reflectCoordinate(coordinate, begin, end);
}

template <typename Sample, int Components>
const std::byte* address(ImageView image, int x, int y) {
    if (x < image.bounds.x1 || x >= image.bounds.x2
        || y < image.bounds.y1 || y >= image.bounds.y2) {
        return nullptr;
    }
    return static_cast<const std::byte*>(image.data)
        + (static_cast<std::ptrdiff_t>(y) - image.bounds.y1) * image.rowBytes
        + (static_cast<std::ptrdiff_t>(x) - image.bounds.x1) * (Components * sizeof(Sample));
}

template <typename Sample>
constexpr float alphaScale() {
    if constexpr (std::is_same_v<Sample, float>) {
        return 1.0f;
    } else {
        return 1.0f / static_cast<float>(std::numeric_limits<Sample>::max());
    }
}

template <typename Sample>
float loadSample(const std::byte* address) {
    Sample component;
    std::memcpy(&component, address, sizeof(Sample));
    return static_cast<float>(component);
}

template <typename Sample, int Components, bool StraightAlpha>
void addPixel(const std::byte* pixel, float weight, std::array<float, Components>& result) {
    if (!pixel || weight == 0.0f) {
        return;
    }
    if constexpr (StraightAlpha) {
        static_assert(Components == 4);
        const float weightedAlpha = weight * loadSample<Sample>(pixel + 3 * sizeof(Sample));
        result[3] += weightedAlpha;
        if (weightedAlpha != 0.0f) {
            const float rgbWeight = weightedAlpha * alphaScale<Sample>();
            for (int channel = 0; channel < 3; ++channel) {
                result[channel] += rgbWeight * loadSample<Sample>(pixel + channel * sizeof(Sample));
            }
        }
    } else {
        for (int channel = 0; channel < Components; ++channel) {
            result[channel] += weight * loadSample<Sample>(pixel + channel * sizeof(Sample));
        }
    }
}

template <typename Sample, int Components, bool StraightAlpha>
void addBilinear(ImageView source, Rect frame, Edge edge, double x, double y,
                 float weight, std::array<float, Components>& result) {
    x = edgeCoordinate(x, frame.x1, frame.x2, edge);
    y = edgeCoordinate(y, frame.y1, frame.y2, edge);
    if (!(x > static_cast<double>(source.bounds.x1) - 1.0 && x < source.bounds.x2
          && y > static_cast<double>(source.bounds.y1) - 1.0 && y < source.bounds.y2)) {
        return;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const auto fx = static_cast<float>(x - x0);
    const auto fy = static_cast<float>(y - y0);

    addPixel<Sample, Components, StraightAlpha>(address<Sample, Components>(source, x0, y0),
                                                (1.0f - fx) * (1.0f - fy) * weight, result);
    addPixel<Sample, Components, StraightAlpha>(address<Sample, Components>(source, x0 + 1, y0),
                                                fx * (1.0f - fy) * weight, result);
    addPixel<Sample, Components, StraightAlpha>(address<Sample, Components>(source, x0, y0 + 1),
                                                (1.0f - fx) * fy * weight, result);
    addPixel<Sample, Components, StraightAlpha>(address<Sample, Components>(source, x0 + 1, y0 + 1),
                                                fx * fy * weight, result);
}

template <typename Sample>
void storeSample(std::byte* destination, float value) {
    Sample component;
    if constexpr (std::is_same_v<Sample, float>) {
        component = value;
    } else {
        const float maxValue = static_cast<float>(std::numeric_limits<Sample>::max());
        component = static_cast<Sample>(std::clamp(value, 0.0f, maxValue) + 0.5f);
    }
    std::memcpy(destination, &component, sizeof(Sample));
}

template <typename Sample, int Components, bool StraightAlpha = false>
void render(ImageView source, MutableImageView destination, Rect frame, Rect window,
            const std::array<Transform, kMaxSamples>& transforms, int samples, Edge edge,
            bool (*abort)(void*), void* abortContext) {
    constexpr int kPixelBytes = Components * sizeof(Sample);
    const float weight = 1.0f / static_cast<float>(samples);
    const Rect available = intersect(source.bounds, frame);
    const bool clear = !source.data || isEmpty(available);
    if (!clear) {
        source.data = address<Sample, Components>(source, available.x1, available.y1);
        source.bounds = available;
    }
    const auto windowBytes = static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(window.x2) - window.x1) * kPixelBytes;
    parallelBands(window.y1, window.y2, [&](int firstRow, int lastRow) {
        for (int y = firstRow; y < lastRow; ++y) {
            if (abort && abort(abortContext)) {
                return;
            }
            auto* output = static_cast<std::byte*>(destination.data)
                + (static_cast<std::ptrdiff_t>(y) - destination.bounds.y1) * destination.rowBytes
                + (static_cast<std::ptrdiff_t>(window.x1) - destination.bounds.x1) * kPixelBytes;
            if (clear) {
                std::memset(output, 0, windowBytes);
                continue;
            }
            for (int x = window.x1; x < window.x2; ++x, output += kPixelBytes) {
                std::array<float, Components> result{};
                for (int sample = 0; sample < samples; ++sample) {
                    const Transform& t = transforms[sample];
                    addBilinear<Sample, Components, StraightAlpha>(
                        source, frame, edge, t.xx * x + t.xy * y + t.xOffset,
                        t.yx * x + t.yy * y + t.yOffset, weight, result);
                }
                if constexpr (StraightAlpha) {
                    // Filter premultiplied RGB through both the bilinear taps and
                    // shutter samples, then restore the host's straight convention.
                    const float alpha = result[3] * alphaScale<Sample>();
                    for (int channel = 0; channel < 3; ++channel) {
                        result[channel] = alpha != 0.0f ? result[channel] / alpha : 0.0f;
                    }
                }
                for (int channel = 0; channel < Components; ++channel) {
                    storeSample<Sample>(output + channel * sizeof(Sample), result[channel]);
                }
            }
        }
    });
}

template <typename Sample>
void renderComponents(ImageView source, MutableImageView destination, Rect frame, Rect window,
                      const std::array<Transform, kMaxSamples>& transforms, int samples,
                      int components, bool straightAlpha, Edge edge, bool (*abort)(void*),
                      void* abortContext) {
    switch (components) {
        case 1:
            render<Sample, 1>(source, destination, frame, window, transforms, samples,
                              edge, abort, abortContext);
            break;
        case 3:
            render<Sample, 3>(source, destination, frame, window, transforms, samples,
                              edge, abort, abortContext);
            break;
        case 4:
            if (straightAlpha) {
                render<Sample, 4, true>(source, destination, frame, window, transforms, samples,
                                       edge, abort, abortContext);
            } else {
                render<Sample, 4>(source, destination, frame, window, transforms, samples,
                                  edge, abort, abortContext);
            }
            break;
    }
}

}  // namespace

void renderShake(
    ImageView source, MutableImageView destination, Rect frame, Rect window,
    Depth depth, int components, double timeSeconds, double frameDurationSeconds,
    double pixelAspect, bool straightAlpha, const ShakeSettings& settings, bool (*abort)(void*),
    void* abortContext) {
    window = intersect(window, destination.bounds);
    if (!destination.data || isEmpty(window) || isEmpty(frame)) {
        return;
    }
    if (!std::isfinite(pixelAspect) || pixelAspect <= 0.0) {
        pixelAspect = 1.0;
    }

    const double shutterTime = std::clamp(settings.shutterDegrees, 0.0, 360.0) / 360.0
        * frameDurationSeconds;
    const int samples = shutterTime > 0.0 && settings.frequency > 0.0 && settings.amplitude > 0.0
        ? std::clamp(settings.samples, 1, kMaxSamples) : 1;
    std::array<Transform, kMaxSamples> transforms{};
    for (int sample = 0; sample < samples; ++sample) {
        const double exposure = (static_cast<double>(sample) + 0.5) / samples - 0.5;
        transforms[sample] = makeTransform(frame, timeSeconds + shutterTime * exposure,
                                           pixelAspect, settings);
    }

    // Rows render on several threads, and only this one may ask the host.
    Cancellation cancellation(abort, abortContext);
    abort = Cancellation::poll;
    abortContext = &cancellation;

    switch (depth) {
        case Depth::Byte:
            renderComponents<std::uint8_t>(source, destination, frame, window, transforms,
                                           samples, components, straightAlpha, settings.edge,
                                           abort, abortContext);
            break;
        case Depth::Short:
            renderComponents<std::uint16_t>(source, destination, frame, window, transforms,
                                            samples, components, straightAlpha, settings.edge,
                                            abort, abortContext);
            break;
        case Depth::Float:
            renderComponents<float>(source, destination, frame, window, transforms,
                                    samples, components, straightAlpha, settings.edge,
                                    abort, abortContext);
            break;
    }
}

}  // namespace diamond
