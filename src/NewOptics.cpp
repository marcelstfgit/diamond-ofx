#include "NewOptics.h"

#include "Effects.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace diamond::fx {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kToRadians = kPi / 180.0;

struct Color {
    float r = 0, g = 0, b = 0;
};

void add(Color& color, Color tint, float light) {
    color.r += tint.r * light;
    color.g += tint.g * light;
    color.b += tint.b * light;
}

Pixel composite(Pixel base, Color light, int blend) {
    const float coverage = std::clamp(std::max({light.r, light.g, light.b}), 0.0f, 1.0f);
    const auto channel = [blend](float source, float emission) {
        if (blend != 1) {
            return source + emission;
        }
        return source + std::min(emission, 1.0f) * (1.0f - std::clamp(source, 0.0f, 1.0f));
    };
    return {channel(base.r, light.r), channel(base.g, light.g), channel(base.b, light.b),
            base.a + (1.0f - base.a) * coverage};
}

template <typename Light>
void renderLight(const Context& context, bool only, Color tint, int blend, const Light& lightAt) {
    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            const Pixel base = only ? Pixel{} : context.source.read(x, y);
            const Color light = lightAt(x, y);
            context.output.pixel(x, y) = composite(
                base, {tint.r * light.r, tint.g * light.g, tint.b * light.b}, only ? 0 : blend);
        }
    });
}

Color gain(double intensity, double red, double green, double blue) {
    return {static_cast<float>(intensity * red), static_cast<float>(intensity * green),
            static_cast<float>(intensity * blue)};
}

float smoothStep(float from, float to, float value) {
    const float t = std::clamp((value - from) / (to - from), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

class Field {
public:
    Field(int width, int height)
        : width_(width), height_(height), values_(static_cast<std::size_t>(width) * height) {}

    int width() const { return width_; }
    int height() const { return height_; }

    float& at(int x, int y) { return values_[index(x, y)]; }
    float at(int x, int y) const { return values_[index(x, y)]; }

    float read(int x, int y) const {
        return x >= 0 && x < width_ && y >= 0 && y < height_ ? at(x, y) : 0.0f;
    }

    float sample(double x, double y) const {
        if (!(x > -1.0 && x < width_ && y > -1.0 && y < height_)) {
            return 0.0f;
        }
        const int left = static_cast<int>(std::floor(x));
        const int bottom = static_cast<int>(std::floor(y));
        const float fx = static_cast<float>(x - left);
        const float fy = static_cast<float>(y - bottom);
        if (left >= 0 && left + 1 < width_ && bottom >= 0 && bottom + 1 < height_) {
            const std::size_t offset = index(left, bottom);
            const float lower = values_[offset] * (1 - fx) + values_[offset + 1] * fx;
            const float upper = values_[offset + width_] * (1 - fx)
                + values_[offset + width_ + 1] * fx;
            return lower * (1 - fy) + upper * fy;
        }
        const float lower = read(left, bottom) * (1 - fx) + read(left + 1, bottom) * fx;
        const float upper = read(left, bottom + 1) * (1 - fx) + read(left + 1, bottom + 1) * fx;
        return lower * (1 - fy) + upper * fy;
    }

private:
    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y) * width_ + x;
    }

    int width_;
    int height_;
    std::vector<float> values_;
};

struct Highlights {
    Highlights(Rect area, double pixelAspect)
        : frame(area), step(std::max(1, (height(area) + 539) / 540)),
          aspect(std::max(0.001, pixelAspect)),
          field((width(area) + step - 1) / step, (height(area) + step - 1) / step) {}

    bool extract(const Context& context, double threshold, double knee, float peakWeight) {
        const float cutoff = static_cast<float>(threshold);
        const float softness = static_cast<float>(knee);
        forEachRow(context, 0, field.height(), [&](int gy) {
            const int y0 = frame.y1 + gy * step;
            const int y1 = std::min(frame.y2, y0 + step);
            for (int gx = 0; gx < field.width(); ++gx) {
                const int x0 = frame.x1 + gx * step;
                const int x1 = std::min(frame.x2, x0 + step);
                float sum = 0.0f;
                float peak = 0.0f;
                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        const Pixel p = context.source.read(x, y);
                        const float luma = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
                        const float excess = std::max(0.0f, luma - cutoff);
                        const float weight = softness > 0.0f
                            ? smoothStep(0.0f, softness, excess) : 1.0f;
                        const float signal = excess * weight;
                        peak = std::max(peak, signal);
                        sum += signal;
                    }
                }
                const int pixels = (x1 - x0) * (y1 - y0);
                field.at(gx, gy) = peak * peakWeight + (sum / pixels) * (1 - peakWeight);
            }
        });
        return !context.cancelled();
    }

    double x(int px) const { return (static_cast<double>(px) - frame.x1 + 0.5) / step - 0.5; }
    double y(int py) const { return (static_cast<double>(py) - frame.y1 + 0.5) / step - 0.5; }
    double pixels(double radiusPercent) const { return height(frame) * radiusPercent / (100.0 * step); }

    Rect frame;
    int step;
    double aspect;
    Field field;
};

// A one-sided exponential convolution follows a ray by reusing the response
// on the preceding major-axis column or row. Runtime is independent of length.
bool scatterRay(const Context& context, const Field& source, Field& scratch, Field& sum,
                double dx, double dy, double length, float power) {
    const bool alongX = std::abs(dx) > std::abs(dy);
    const double main = alongX ? dx : dy;
    const double minor = alongX ? dy : dx;
    const int scanLength = alongX ? source.width() : source.height();
    const int scanWidth = alongX ? source.height() : source.width();
    const int direction = main > 0.0 ? 1 : -1;
    const int side = minor > 0.0 ? 1 : -1;
    const float fraction = static_cast<float>(std::abs(minor / main));
    const auto falloff = static_cast<float>(std::exp(-1.0 / std::max(0.05, length * std::abs(main))));
    const float inputWeight = std::max(1.0f - falloff, 0.12f);
    for (int depth = 0; depth < scanLength; ++depth) {
        if (context.cancelled()) {
            return false;
        }
        const int major = direction == 1 ? depth : scanLength - 1 - depth;
        const int previous = major - direction;
        for (int j = 0; j < scanWidth; ++j) {
            const int x = alongX ? major : j;
            const int y = alongX ? j : major;
            const bool validNeighbor = j - side >= 0 && j - side < scanWidth;
            float ray = 0.0f;
            if (depth > 0) {
                const float straight = alongX ? scratch.at(previous, j) : scratch.at(j, previous);
                float shifted = 0.0f;
                if (validNeighbor) {
                    shifted = alongX ? scratch.at(previous, j - side) : scratch.at(j - side, previous);
                }
                ray = falloff * (straight * (1 - fraction) + shifted * fraction);
            }
            scratch.at(x, y) = source.at(x, y) * inputWeight + ray;
            sum.at(x, y) += power * ray;
        }
    }
    return true;
}

bool boxBlur(const Context& context, const Field& source, Field& output, int radiusX, int radiusY) {
    Field intermediate(source.width(), source.height());
    const float inverseX = 1.0f / (2 * radiusX + 1);
    const float inverseY = 1.0f / (2 * radiusY + 1);
    for (int y = 0; y < source.height(); ++y) {
        if (context.cancelled()) {
            return false;
        }
        float sum = 0;
        for (int x = -radiusX; x <= radiusX; ++x) {
            sum += source.read(x, y);
        }
        for (int x = 0; x < source.width(); ++x) {
            intermediate.at(x, y) = std::max(0.0f, sum * inverseX);
            sum += source.read(x + radiusX + 1, y) - source.read(x - radiusX, y);
        }
    }
    for (int x = 0; x < source.width(); ++x) {
        if (context.cancelled()) {
            return false;
        }
        float sum = 0;
        for (int y = -radiusY; y <= radiusY; ++y) {
            sum += intermediate.read(x, y);
        }
        for (int y = 0; y < source.height(); ++y) {
            output.at(x, y) = std::max(0.0f, sum * inverseY);
            sum += intermediate.read(x, y + radiusY + 1) - intermediate.read(x, y - radiusY);
        }
    }
    return true;
}

bool renderDisabled(const Context& context, double intensity, double radius, bool only) {
    if (intensity > 0 && radius > 0 && width(context.frame) > 0 && height(context.frame) > 0) {
        return false;
    }
    renderLight(context, only, {}, 0, [](int, int) { return Color{}; });
    return true;
}

float shoulder(float value) { return value / (1.0f + 0.34f * value); }

std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float noise(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    const auto index = static_cast<std::uint32_t>(x) * 0x85ebca6bU
        ^ static_cast<std::uint32_t>(y) * 0x27d4eb2fU;
    return static_cast<float>(hash(index ^ seed)) / 2147483648.0f - 1.0f;
}

double wrapPhase(double phase) {
    return std::isfinite(phase) ? std::remainder(phase, 65536.0) : 0.0;
}

float animatedNoise(double position, std::uint32_t seed) {
    const double cell = std::floor(position);
    const auto index = static_cast<std::int32_t>(cell);
    const float t = smoothStep(0, 1, static_cast<float>(position - cell));
    const float first = noise(index, 0, seed);
    return first + (noise(index + 1, 0, seed) - first) * t;
}

struct BurnPalette {
    Color edge, middle, hot;
    double stretch;
};

constexpr std::array<BurnPalette, 3> kBurnPalettes{{
    {{0.99f, 0.12f, 0.022f}, {1.0f, 0.58f, 0.06f}, {1.0f, 0.97f, 0.77f}, 1.0},
    {{0.83f, 0.018f, 0.22f}, {1.0f, 0.25f, 0.20f}, {1.0f, 0.92f, 0.77f}, 1.22},
    {{0.89f, 0.47f, 0.25f}, {1.0f, 0.83f, 0.58f}, {1.0f, 0.99f, 0.91f}, 0.76},
}};

} // namespace

void renderGlare(const Context& context, const GlareSettings& settings) {
    if (renderDisabled(context, settings.intensity, settings.radius, settings.glareOnly)) {
        return;
    }
    Highlights input(context.frame, context.pixelAspect);
    if (!input.extract(context, settings.threshold, settings.knee, 0.3f)) {
        return;
    }
    const double size = input.pixels(settings.radius);
    const Field& highlights = input.field;
    Field blurred(highlights.width(), highlights.height());
    Field halo(highlights.width(), highlights.height());
    Field rays(highlights.width(), highlights.height());
    Field scratch(highlights.width(), highlights.height());
    const int blurX = std::max(1, static_cast<int>(std::round(size * 0.09 / input.aspect)));
    const int blurY = std::max(1, static_cast<int>(std::round(size * 0.09)));
    if (!boxBlur(context, highlights, blurred, blurX, blurY)
        || !boxBlur(context, blurred, halo, blurX, blurY)) {
        return;
    }
    const int blades = std::clamp(settings.blades, 2, 12);
    for (int blade = 0; blade < blades; ++blade) {
        const double angle = settings.rotation * kToRadians + 2.0 * kPi * blade / blades;
        if (!scatterRay(context, blurred, scratch, rays, std::cos(angle) / input.aspect,
                        std::sin(angle), size * 0.34, 1.35f / blades)) {
            return;
        }
    }

    // Separated ring radii make the chromatic fringe follow each highlight.
    struct RingPoint { double x, y; };
    constexpr int kRingSamples = 32;
    std::array<RingPoint, kRingSamples> redRing{}, greenRing{}, blueRing{};
    const double dispersion = 0.22 * std::clamp(settings.dispersion, 0.0, 1.0);
    for (int index = 0; index < kRingSamples; ++index) {
        const double angle = 2.0 * kPi * index / kRingSamples;
        const double x = std::cos(angle) * size / input.aspect;
        const double y = std::sin(angle) * size;
        redRing[index] = {x * (0.72 + dispersion), y * (0.72 + dispersion)};
        greenRing[index] = {x * 0.72, y * 0.72};
        blueRing[index] = {x * (0.72 - dispersion), y * (0.72 - dispersion)};
    }
    Field red(highlights.width(), highlights.height());
    Field green(highlights.width(), highlights.height());
    Field blue(highlights.width(), highlights.height());
    forEachRow(context, 0, highlights.height(), [&](int y) {
        for (int x = 0; x < highlights.width(); ++x) {
            const float broad = 1.24f * halo.at(x, y) + 0.38f * rays.at(x, y);
            Color light{broad, broad, broad};
            for (int tap = 0; tap < kRingSamples; ++tap) {
                const RingPoint& r = redRing[tap];
                const RingPoint& g = greenRing[tap];
                const RingPoint& b = blueRing[tap];
                light.r += 0.27f * halo.sample(x + r.x, y + r.y);
                light.g += 0.27f * halo.sample(x + g.x, y + g.y);
                light.b += 0.27f * halo.sample(x + b.x, y + b.y);
            }
            red.at(x, y) = shoulder(light.r);
            green.at(x, y) = shoulder(light.g);
            blue.at(x, y) = shoulder(light.b);
        }
    });
    if (context.cancelled()) {
        return;
    }
    const Color tint = gain(settings.intensity, settings.red, settings.green, settings.blue);
    renderLight(context, settings.glareOnly, tint, settings.blend, [&](int x, int y) {
        const double gx = input.x(x);
        const double gy = input.y(y);
        return Color{red.sample(gx, gy), green.sample(gx, gy), blue.sample(gx, gy)};
    });
}

void renderGlint(const Context& context, const GlintSettings& settings) {
    if (renderDisabled(context, settings.intensity, settings.length, settings.glintOnly)) {
        return;
    }
    Highlights input(context.frame, context.pixelAspect);
    if (!input.extract(context, settings.threshold, settings.knee, 0.77f)) {
        return;
    }
    const double size = input.pixels(settings.length);
    const Field& highlights = input.field;
    Field rays(highlights.width(), highlights.height());
    Field scratch(highlights.width(), highlights.height());
    const int arms = std::clamp(settings.rays, 2, 12);
    const double angle = settings.angle * kToRadians;
    for (int index = 0; index < arms; ++index) {
        const double direction = angle + 2.0 * kPi * index / arms;
        const bool prominent = (index % std::max(1, arms / 2)) == 0;
        const double length = size * (prominent ? 0.48 : 0.26);
        const float power = (prominent ? 2.5f : 1.6f) / std::sqrt(static_cast<float>(arms));
        if (!scatterRay(context, highlights, scratch, rays, std::cos(direction) / input.aspect,
                        std::sin(direction), length, power)) {
            return;
        }
    }
    const double slit = 2.0 * std::clamp(settings.dispersion, 0.0, 1.0);
    const double offsetX = -std::sin(angle) * slit / input.aspect;
    const double offsetY = std::cos(angle) * slit;
    const Color tint = gain(settings.intensity, settings.red, settings.green, settings.blue);
    renderLight(context, settings.glintOnly, tint, settings.blend, [&](int x, int y) {
        const double gx = input.x(x);
        const double gy = input.y(y);
        const float center = shoulder(rays.sample(gx, gy));
        const float emitter = 0.36f * shoulder(highlights.sample(gx, gy));
        const float redFringe = shoulder(rays.sample(gx + offsetX, gy + offsetY));
        const float blueFringe = shoulder(rays.sample(gx - offsetX, gy - offsetY));
        return Color{emitter + 0.67f * center + 0.33f * redFringe,
                     emitter + center,
                     emitter + 0.67f * center + 0.33f * blueFringe};
    });
}

void renderStreaks(const Context& context, const StreaksSettings& settings) {
    if (renderDisabled(context, settings.intensity, settings.length, settings.streaksOnly)) {
        return;
    }
    Highlights input(context.frame, context.pixelAspect);
    if (!input.extract(context, settings.threshold, settings.knee, 0.61f)) {
        return;
    }
    const double size = input.pixels(settings.length);
    const Field& highlights = input.field;
    Field trails(highlights.width(), highlights.height());
    Field scratch(highlights.width(), highlights.height());
    const int count = std::clamp(settings.count, 1, 8);
    for (int index = 0; index < count; ++index) {
        const double fan = count > 1 ? 2.0 * index / (count - 1) - 1.0 : 0.0;
        const double direction = settings.angle * kToRadians + fan * settings.spread * (kPi / 5.5);
        const double dx = std::cos(direction) / input.aspect;
        const double dy = std::sin(direction);
        const double length = size * (0.46 - 0.12 * std::abs(fan));
        const float power = 1.52f / static_cast<float>(count);
        if (!scatterRay(context, highlights, scratch, trails, dx, dy, length, power)
            || !scatterRay(context, highlights, scratch, trails, -dx, -dy, length * 0.78,
                           power * 0.81f)) {
            return;
        }
    }
    const Color tint = gain(settings.intensity, settings.red, settings.green, settings.blue);
    renderLight(context, settings.streaksOnly, tint, settings.blend, [&](int x, int y) {
        const float light = shoulder(trails.sample(input.x(x), input.y(y)));
        return Color{light, light, light};
    });
}

void renderFilmBurn(const Context& context, const FilmBurnSettings& settings) {
    if (renderDisabled(context, settings.amount, settings.size, settings.burnOnly)) {
        return;
    }
    const int style = std::clamp(settings.style, 0, static_cast<int>(kBurnPalettes.size()) - 1);
    const BurnPalette& palette = kBurnPalettes[style];
    const auto seed = static_cast<std::uint32_t>(settings.seed);
    const double aspect = std::max(0.001, context.pixelAspect);
    const double fps = context.fps > 0.0 ? context.fps : 30.0;
    const double phase = wrapPhase(context.timeFrames * settings.speed / fps);
    const double nominalSize = height(context.frame) * std::max(0.05, settings.size);
    const double motionX = 0.24 * animatedNoise(phase, seed ^ 0xb950b44bU);
    const double motionY = 0.18 * animatedNoise(phase * 0.83, seed ^ 0xf07799c9U);
    const double centerX = (context.frame.x1 + settings.centerX * (width(context.frame) - 1.0))
        * aspect + motionX * nominalSize;
    const double centerY = context.frame.y1 + settings.centerY * (height(context.frame) - 1.0)
        + motionY * nominalSize;
    const double angle = settings.angle * kToRadians;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double shapeA = 3.2 * animatedNoise(phase * 1.4, seed ^ 0xe1d63d39U);
    const double shapeB = 3.2 * animatedNoise(phase * 0.7, seed ^ 0x1e8b3319U);
    const double pulse = 0.78 + 0.36 * animatedNoise(phase * 2.4, seed ^ 0x2eb7e921U);
    const double softness = std::clamp(settings.softness, 0.0, 1.0);
    const auto grainSeed = seed ^ hash(static_cast<std::uint32_t>(
        static_cast<std::int32_t>(std::floor(wrapPhase(phase * 24))))) ^ 0x5a4e7713U;
    const Color tint = gain(settings.amount, settings.red, settings.green, settings.blue);
    renderLight(context, settings.burnOnly, tint, settings.blend, [&](int x, int y) {
        const double dx = (x * aspect - centerX) / nominalSize;
        const double dy = (y - centerY) / nominalSize;
        const double u = cosine * dx + sine * dy;
        const double v = (cosine * dy - sine * dx) / palette.stretch;
        // The exposure front has a ragged organic boundary and a warm-to-white
        // response. Smaller waves disappear into the falloff at coarse scales.
        const double distorted = u + 0.42 * v * v + 0.09 * std::sin(v * 7.4 + shapeA)
            + 0.032 * std::sin(v * 19.5 + u * 11.0 + shapeB)
            + 0.010 * std::sin(v * 57.0 - u * 39.0 + shapeA);
        const double du = distorted / (0.09 + softness * 0.31);
        const double dv = (v + 0.13 * distorted) / (0.34 + softness * 0.54);
        const auto distance = static_cast<float>(du * du + dv * dv);
        const float fringe = 1.0f - smoothStep(0.53f, 3.8f, distance);
        const float body = 1.0f - smoothStep(0.20f, 1.35f, distance);
        const float core = 1.0f - smoothStep(0.015f, 0.44f, distance);
        if (fringe <= 0.0f) {
            return Color{};
        }
        const float variation = 0.93f + 0.07f * noise(x, y, grainSeed);
        const auto strength = static_cast<float>(pulse) * variation;
        Color light{};
        add(light, palette.edge, 0.86f * fringe * strength);
        add(light, palette.middle, 1.02f * body * strength);
        add(light, palette.hot, 1.64f * core * strength);
        return light;
    });
}

} // namespace diamond::fx
