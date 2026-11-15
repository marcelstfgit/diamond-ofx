#include "NewStylize.h"

#include <array>
#include <cstdint>
#include <limits>

namespace diamond::fx {
namespace {

constexpr double pi = 3.14159265358979323846;

struct Color {
    float red;
    float green;
    float blue;
};

Color unpremultiply(Pixel value)
{
    if (!(value.a > 1.0e-6f)) {
        return {};
    }
    return {value.r / value.a, value.g / value.a, value.b / value.a};
}

Pixel premultiply(Color color, float alpha)
{
    return {color.red * alpha, color.green * alpha, color.blue * alpha, alpha};
}

float luma(Color color)
{
    return 0.2126f * color.red + 0.7152f * color.green + 0.0722f * color.blue;
}

float lerp(float first, float second, float mix)
{
    return first + (second - first) * mix;
}

Pixel blend(Pixel first, Pixel second, float mix)
{
    if (mix >= 1.0f) {
        return second;
    }
    if (mix <= 0.0f) {
        return first;
    }
    return {lerp(first.r, second.r, mix), lerp(first.g, second.g, mix), lerp(first.b, second.b, mix),
        lerp(first.a, second.a, mix)};
}

float unit(double value, float fallback = 0.0f)
{
    return std::isfinite(value) ? static_cast<float>(std::clamp(value, 0.0, 1.0)) : fallback;
}

double bounded(double value, double low, double high, double fallback)
{
    return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}

float smooth(float value)
{
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

template<class Evaluate>
void draw(const Context& context, const Evaluate& evaluate)
{
    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            context.output.pixel(x, y) = evaluate(x, y, context.source.read(x, y));
        }
    });
}

void copySource(const Context& context)
{
    draw(context, [](int, int, Pixel pixel) { return pixel; });
}

std::uint32_t hash(std::uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float randomAt(std::uint32_t seed, int x, int y)
{
    const auto bits = hash(seed ^ (0x9e3779b9u * static_cast<std::uint32_t>(x))
        ^ (0x85ebca6bu * static_cast<std::uint32_t>(y)));
    return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

double heldTime(const Context& context, double frequency)
{
    const double frameRate = bounded(context.fps, 1.0, 1000.0, 30.0);
    const double seconds = std::isfinite(context.timeFrames) ? context.timeFrames / frameRate : 0.0;
    return frequency > 0.0 ? std::floor(std::fmod(seconds, 86400.0) * frequency) / frequency : 0.0;
}

struct Chroma {
    float redDifference;
    float blueDifference;
};

Chroma chroma(Color color, float brightness)
{
    return {color.red - brightness, color.blue - brightness};
}

Chroma sampleChroma(Pixel sample, Chroma fallback)
{
    if (!(sample.a > 1.0e-6f)) {
        return fallback;
    }
    const auto color = unpremultiply(sample);
    return chroma(color, luma(color));
}

Color fromLumaChroma(float brightness, Chroma color)
{
    return {brightness + color.redDifference,
        brightness - (0.2126f * color.redDifference + 0.0722f * color.blueDifference) / 0.7152f,
        brightness + color.blueDifference};
}

float printDensity(float channel, float threshold, float contrast)
{
    return unit(0.5 + (threshold - channel) * contrast);
}

// Area of a circular dot clipped to a square cell of unit width.
double dotArea(double radius)
{
    const double squared = radius * radius;
    if (radius <= 0.5) {
        return pi * squared;
    }
    const double segment = squared * std::acos(0.5 / radius) - 0.5 * std::sqrt(squared - 0.25);
    return pi * squared - 4.0 * segment;
}

const std::array<float, 257>& dotRadii()
{
    static const auto table = [] {
        std::array<float, 257> result{};
        for (std::size_t index = 1; index < result.size(); ++index) {
            const double target = static_cast<double>(index) / (result.size() - 1);
            double low = 0.0;
            double high = std::sqrt(0.5);
            for (int step = 0; step < 16; ++step) {
                const double middle = (low + high) * 0.5;
                if (dotArea(middle) < target) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            result[index] = static_cast<float>((low + high) * 0.5);
        }
        return result;
    }();
    return table;
}

struct Screen {
    double cosine;
    double sine;
    double cell;
    double aspect;
    double originX;
    double originY;
    float threshold;
    float contrast;
    float softness;
    bool invert;

    std::array<double, 2> grid(double x, double y) const
    {
        const double horizontal = (x - originX) * aspect / cell;
        const double vertical = (y - originY) / cell;
        return {cosine * horizontal + sine * vertical, -sine * horizontal + cosine * vertical};
    }

    Pixel readGrid(const Image& image, double u, double v) const
    {
        const double x = originX + (cosine * u - sine * v) * cell / aspect;
        const double y = originY + (sine * u + cosine * v) * cell;
        return image.sample(x, y, Edge::Clamp);
    }

    float density(Color source, int plate) const
    {
        if (invert) {
            source = {1.0f - source.red, 1.0f - source.green, 1.0f - source.blue};
        }
        if (plate == 0) {
            return printDensity(luma(source), threshold, contrast);
        }

        const Color ink = {printDensity(source.red, threshold, contrast), printDensity(source.green, threshold, contrast),
            printDensity(source.blue, threshold, contrast)};
        if (plate == 1) {
            return ink.red;
        }
        if (plate == 2) {
            return ink.green;
        }
        if (plate == 3) {
            return ink.blue;
        }

        // Remove the shared neutral component from CMY and print that density on the K plate.
        const float black = std::min({ink.red, ink.green, ink.blue});
        if (plate == 7) {
            return black;
        }
        const std::array<float, 3> chromatic{ink.red, ink.green, ink.blue};
        return (chromatic[static_cast<std::size_t>(plate - 4)] - black) / std::max(1.0e-6f, 1.0f - black);
    }

    Color cellColor(const Image& image, double u, double v) const
    {
        // Average premultiplied samples inside the cell so transparent pixels cannot stain its ink.
        constexpr double offset = 0.23;
        Pixel average{};
        for (const double dy : {-offset, offset}) {
            for (const double dx : {-offset, offset}) {
                const auto sample = readGrid(image, u + dx, v + dy);
                average.r += sample.r;
                average.g += sample.g;
                average.b += sample.b;
                average.a += sample.a;
            }
        }
        return unpremultiply(average);
    }
};

float mark(float area, double u, double v, float blur)
{
    if (area <= 0.0f || area >= 1.0f) {
        return area;
    }
    const float index = area * 256.0f;
    const auto lower = static_cast<std::size_t>(std::min(index, 255.0f));
    const auto& lookup = dotRadii();
    const float radius = lerp(lookup[lower], lookup[lower + 1], index - static_cast<float>(lower));
    const float distance = static_cast<float>(std::sqrt(u * u + v * v));
    return smooth((radius - distance + blur) / (2.0f * blur));
}

template<class StoreInk>
bool renderScreen(const Context& context, const Screen& screen, int plate, const StoreInk& storeInk)
{
    const Rect window = context.output.bounds;
    const double x2 = window.x2 - 1.0;
    const double y2 = window.y2 - 1.0;
    const auto a = screen.grid(window.x1, window.y1);
    const auto b = screen.grid(x2, window.y1);
    const auto c = screen.grid(window.x1, y2);
    const auto d = screen.grid(x2, y2);
    const double narrowestCell = screen.cell / std::max(1.0, screen.aspect);
    if (narrowestCell < 1.25) {
        forEachRow(context, window.y1, window.y2, [&](int y) {
            for (int x = window.x1; x < window.x2; ++x) {
                const auto value = context.source.read(x, y);
                storeInk(context.output.pixel(x, y), screen.density(unpremultiply(value), plate));
            }
        });
        return !context.cancelled();
    }

    const int left = static_cast<int>(std::floor(std::min({a[0], b[0], c[0], d[0]}) + 0.5));
    const int bottom = static_cast<int>(std::floor(std::min({a[1], b[1], c[1], d[1]}) + 0.5));
    const int columns = 1 + static_cast<int>(std::floor(std::max({a[0], b[0], c[0], d[0]}) + 0.5)) - left;
    const int rows = 1 + static_cast<int>(std::floor(std::max({a[1], b[1], c[1], d[1]}) + 0.5)) - bottom;
    // Every cell's ink is worked out before the pixels are drawn, so the rows
    // below only read the table and can be drawn from several threads.
    std::vector<float> densities(static_cast<std::size_t>(columns) * rows);
    forEachRow(context, 0, rows, [&](int row) {
        auto* cells = densities.data() + static_cast<std::size_t>(row) * columns;
        for (int column = 0; column < columns; ++column) {
            cells[column] = screen.density(screen.cellColor(context.source, left + column, bottom + row), plate);
        }
    });
    if (context.cancelled()) {
        return false;
    }
    const float blur = static_cast<float>(std::min(0.30, 0.55 / narrowestCell + 0.16 * screen.softness));
    const float resolve = smooth(static_cast<float>((narrowestCell - 1.25) / 1.25));
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            const auto position = screen.grid(x, y);
            const int nearestU = static_cast<int>(std::floor(position[0] + 0.5));
            const int nearestV = static_cast<int>(std::floor(position[1] + 0.5));
            const float density = densities[static_cast<std::size_t>(nearestV - bottom) * columns + nearestU - left];
            const float ink = mark(density, position[0] - nearestU, position[1] - nearestV, blur);
            storeInk(context.output.pixel(x, y), lerp(density, ink, resolve));
        }
    });
    return !context.cancelled();
}

} // namespace

void renderVHSDamage(const Context& context, const VHSSettings& settings)
{
    const Rect window = context.output.bounds;
    const int frameWidth = width(context.frame);
    const int frameHeight = height(context.frame);
    const float amount = unit(settings.amount);
    if (amount == 0.0f || frameWidth == 0 || frameHeight == 0) {
        copySource(context);
        return;
    }

    const double tracking = unit(settings.tracking);
    const double jitter = unit(settings.jitter);
    const double bleed = unit(settings.chromaBleed);
    const float scanlines = unit(settings.scanlines);
    const float noise = unit(settings.noise);
    const float dropout = unit(settings.dropout);
    if (tracking == 0.0 && jitter == 0.0 && bleed == 0.0 && scanlines == 0.0f && noise == 0.0f && dropout == 0.0f) {
        copySource(context);
        return;
    }
    const double frequency = bounded(settings.frequency, 0.0, 60.0, 24.0);
    const double time = heldTime(context, frequency);
    const auto timeKey = static_cast<std::uint32_t>(static_cast<std::int32_t>(std::llround(time * frequency)));
    const auto seed = hash(hash(timeKey) ^ static_cast<std::uint32_t>(settings.seed) ^ 0x88f465adu);
    const double widthScale = frameWidth / 1920.0;
    const double lineSpacing = std::max(1.0, frameHeight / 480.0);
    const int stripes = 10 + static_cast<int>(40 * dropout);
    const double headHeight = 0.035 + 0.095 * tracking;
    const double headOffset = (randomAt(seed, 0, 1) - 0.38) * 60.0 * widthScale * tracking;
    const double tearCenter = 0.22 + 0.66 * (0.5 + 0.5 * std::sin(time * 3.0 + 0.8));
    const double tearOffset = (2.0 * randomAt(seed, 0, 2) - 1.0) * widthScale * 75.0 * tracking;
    const double bleedDistance = widthScale * bleed * 9.0;
    const int stripeWidth = std::max(1, frameWidth / stripes);
    const int streakGrid = std::max(1, static_cast<int>(std::round(widthScale * 2.0)));
    const double fieldPhase = (timeKey & 1u) == 0 ? 1.0 : -1.0;

    forEachRow(context, window.y1, window.y2, [&](int y) {
        const double row = (static_cast<double>(y) - context.frame.y1) / frameHeight;
        const double scanPosition = row * frameHeight / lineSpacing;
        const int line = static_cast<int>(std::floor(scanPosition));
        const double lineRoll = 2.0 * randomAt(seed, 1, line) - 1.0;
        const double headWeight = 1.0 - smooth(static_cast<float>(row / headHeight));
        const double tearWeight = 1.0 - smooth(static_cast<float>(std::abs(row - tearCenter) / 0.032));
        const double ripple = std::sin(row * 39.0 + time * 9.0) + 0.45 * std::sin(row * 105.0 - time * 5.0);
        const double fieldOffset = (line % 2 == 0 ? fieldPhase : -fieldPhase) * 0.12 * scanlines;
        const double shift = (ripple + 0.42 * lineRoll + fieldOffset) * 9.0 * jitter * widthScale
            + tearWeight * tearOffset + headWeight * (headOffset + lineRoll * 28.0 * tracking * widthScale);
        const float carrier = static_cast<float>(0.5 + 0.5 * std::cos(pi * scanPosition));
        const float scanGain = 1.0f - 0.23f * scanlines * carrier;
        const float lineNoise = static_cast<float>(lineRoll) * (0.024f + 0.058f * static_cast<float>(headWeight) * static_cast<float>(tracking));
        int previousStripe = std::numeric_limits<int>::min();
        float dropoutGain = 1.0f;
        float dropoutLift = 0.0f;
        for (int x = window.x1; x < window.x2; ++x) {
            const Pixel original = context.source.read(x, y);
            const auto warped = shift == 0.0 ? original : context.source.sample(x + shift, y, Edge::Clamp);
            if (!(warped.a > 1.0e-6f)) {
                context.output.pixel(x, y) = blend(original, warped, amount);
                continue;
            }
            const auto color = unpremultiply(warped);
            const float originalLuma = luma(color);
            auto colorSignal = chroma(color, originalLuma);
            if (bleedDistance > 0.0) {
                const auto late = sampleChroma(context.source.sample(x + shift - bleedDistance, y, Edge::Clamp), colorSignal);
                const auto latest = sampleChroma(context.source.sample(x + shift - bleedDistance * 2.4, y, Edge::Clamp), colorSignal);
                colorSignal.redDifference = lerp(colorSignal.redDifference, 0.70f * late.redDifference + 0.30f * latest.redDifference,
                    static_cast<float>(0.92 * bleed));
                colorSignal.blueDifference = lerp(colorSignal.blueDifference, 0.52f * late.blueDifference + 0.48f * latest.blueDifference,
                    static_cast<float>(0.92 * bleed));
            }

            const int localX = x - context.frame.x1;
            const int stripe = localX / stripeWidth;
            if (stripe != previousStripe && dropout > 0.0f) {
                previousStripe = stripe;
                const float roll = randomAt(seed ^ 0x60212d4bu, stripe, line / 2);
                const float threshold = dropout * static_cast<float>(0.045 + 0.25 * headWeight * tracking);
                const float dropoutStrength = threshold > 0.0f ? std::max(0.0f, 1.0f - roll / threshold) : 0.0f;
                dropoutGain = 1.0f - 0.44f * dropoutStrength;
                dropoutLift = 0.43f * dropoutStrength;
            }
            float grain = 0.0f;
            if (noise > 0.0f) {
                const float intermittent = randomAt(seed ^ 0xdab9471bu, localX / streakGrid, line) - 0.5f;
                const float fine = randomAt(seed ^ 0x5dd48f37u, localX, y - context.frame.y1) - 0.5f;
                grain = noise * (lineNoise + 0.15f * intermittent + 0.045f * fine);
            }
            const float brightness = originalLuma * scanGain * dropoutGain + grain + dropoutLift;
            context.output.pixel(x, y) = blend(original, premultiply(fromLumaChroma(brightness, colorSignal), warped.a), amount);
        }
    });
}

void renderHalfTone(const Context& context, const HalfToneSettings& settings)
{
    const auto mix = unit(settings.mix);
    const int frameWidth = width(context.frame);
    const int frameHeight = height(context.frame);
    if (mix == 0.0f || frameWidth == 0 || frameHeight == 0) {
        copySource(context);
        return;
    }
    if (width(context.output.bounds) == 0 || height(context.output.bounds) == 0) {
        return;
    }

    const Color ink{unit(settings.inkRed), unit(settings.inkGreen), unit(settings.inkBlue)};
    const Color paper{unit(settings.paperRed, 1.0f), unit(settings.paperGreen, 1.0f), unit(settings.paperBlue, 1.0f)};
    const double angle = bounded(settings.angle, -360.0, 360.0, 0.0);
    const double aspect = bounded(context.pixelAspect, 0.01, 100.0, 1.0);
    const double cell = bounded(settings.cellSize, 2.0, 120.0, 9.0) * frameHeight / 1080.0;
    auto screen = [&](double angleOffset) {
        const double radians = (angle + angleOffset) * pi / 180.0;
        return Screen{std::cos(radians), std::sin(radians), cell, aspect,
            context.frame.x1 + (frameWidth - 1.0) * 0.5, context.frame.y1 + (frameHeight - 1.0) * 0.5,
            unit(settings.threshold, 0.5f), static_cast<float>(bounded(settings.contrast, 0.2, 4.0, 1.0)),
            unit(settings.softness), settings.invert};
    };
    if (settings.mode == 0) {
        if (!renderScreen(context, screen(0.0), 0, [](Pixel& value, float area) { value.r = area; })) {
            return;
        }
    } else {
        constexpr std::array<double, 4> rgbAngles{-15.0, 15.0, 45.0, 0.0};
        constexpr std::array<double, 4> printAngles{-15.0, 15.0, 0.0, 45.0};
        constexpr std::array<float Pixel::*, 4> channels{&Pixel::r, &Pixel::g, &Pixel::b, &Pixel::a};
        const bool cmyk = settings.mode == 2;
        const auto& offsets = cmyk ? printAngles : rgbAngles;
        const int count = cmyk ? 4 : 3;
        for (int channel = 0; channel < count; ++channel) {
            const auto member = channels[static_cast<std::size_t>(channel)];
            if (!renderScreen(context, screen(offsets[static_cast<std::size_t>(channel)]), channel + (cmyk ? 4 : 1),
                [member](Pixel& value, float area) { value.*member = area; })) {
                return;
            }
        }
    }

    draw(context, [&](int x, int y, Pixel original) {
        const auto marked = context.output.pixel(x, y);
        Color density{marked.r, marked.g, marked.b};
        if (settings.mode == 0) {
            density = {marked.r, marked.r, marked.r};
        } else if (settings.mode == 2) {
            density.red = lerp(density.red, 1.0f, marked.a);
            density.green = lerp(density.green, 1.0f, marked.a);
            density.blue = lerp(density.blue, 1.0f, marked.a);
        }
        const Color printed{lerp(paper.red, ink.red, density.red), lerp(paper.green, ink.green, density.green),
            lerp(paper.blue, ink.blue, density.blue)};
        return blend(original, premultiply(printed, original.a), mix);
    });
}

void renderBleachBypass(const Context& context, const BleachBypassSettings& settings)
{
    const float mix = unit(settings.amount) * unit(settings.mix);
    if (mix == 0.0f) {
        copySource(context);
        return;
    }

    const float exposure = static_cast<float>(std::exp2(bounded(settings.exposure, -5.0, 5.0, 0.0)));
    const float contrast = static_cast<float>(bounded(settings.contrast, 0.25, 3.0, 1.0));
    const float desaturation = unit(settings.desaturation);
    const float lift = static_cast<float>(bounded(settings.shadowLift, -0.25, 0.25, 0.0));
    const float shoulder = unit(settings.highlightRolloff) * 0.85f;
    const float temperature = static_cast<float>(bounded(settings.temperature, -1.0, 1.0, 0.0));
    draw(context, [&](int, int, Pixel original) {
        if (!(original.a > 1.0e-6f)) {
            return original;
        }
        auto color = unpremultiply(original);
        color = {color.red * exposure, color.green * exposure, color.blue * exposure};
        const float signal = luma(color);
        constexpr float pivot = 0.41f;
        float brightness = (signal - pivot) * contrast + pivot;
        // The toe protects blacks from the main contrast curve, while the shoulder retains HDR headroom.
        const float shadow = 1.0f - smooth(signal / pivot);
        brightness = lerp(brightness, signal, 0.72f * shadow);
        const float excess = std::max(0.0f, brightness - 0.65f);
        brightness -= shoulder * excess * (excess / (0.32f + excess));
        brightness += lift * shadow * shadow;

        const float silver = 1.0f - desaturation * (0.88f + 0.12f * shadow);
        auto retained = chroma(color, signal);
        retained.redDifference *= silver;
        retained.blueDifference *= silver;
        const float tint = temperature * (std::clamp(brightness, 0.0f, 1.0f) * 0.11f + shadow * 0.024f);
        retained.redDifference += tint;
        retained.blueDifference -= tint * 1.37f;
        return blend(original, premultiply(fromLumaChroma(brightness, retained), original.a), mix);
    });
}

} // namespace diamond::fx
