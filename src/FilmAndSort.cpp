#include "Effects.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace diamond::fx {
namespace {

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

float lerp(float from, float to, float mix)
{
    return from + (to - from) * mix;
}

Pixel blend(Pixel first, Pixel second, float mix)
{
    return {lerp(first.r, second.r, mix), lerp(first.g, second.g, mix),
        lerp(first.b, second.b, mix), lerp(first.a, second.a, mix)};
}

float luma(float red, float green, float blue)
{
    return 0.2126f * red + 0.7152f * green + 0.0722f * blue;
}

float smoothUnit(float value)
{
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

void copySource(const Context& context)
{
    const Rect bounds = context.output.bounds;
    forEachRow(context, bounds.y1, bounds.y2, [&](int y) {
        for (int x = bounds.x1; x < bounds.x2; ++x) {
            context.output.pixel(x, y) = context.source.read(x, y);
        }
    });
}

struct SortValue {
    Pixel pixel;
    float key;
    int index;
};

float sortKey(int mode, Pixel pixel, float luminance)
{
    switch (mode) {
    case 1: return pixel.r / pixel.a;
    case 2: return pixel.g / pixel.a;
    case 3: return pixel.b / pixel.a;
    case 4: {
        const float low = std::min({pixel.r, pixel.g, pixel.b});
        const float high = std::max({pixel.r, pixel.g, pixel.b});
        return high > 0.0f ? std::clamp((high - low) / high, 0.0f, 1.0f) : 0.0f;
    }
    default: return luminance;
    }
}

struct Scanline {
    bool horizontalMajor;
    int intercept;
    int slopeSign;
    const std::vector<int>& offsets;
    Rect frame;

    void position(int index, int& x, int& y) const
    {
        const int minor = intercept + slopeSign * offsets[static_cast<std::size_t>(index)];
        x = frame.x1 + (horizontalMajor ? index : minor);
        y = frame.y1 + (horizontalMajor ? minor : index);
    }
};

void writeSortPixel(const Context& context, const Scanline& line, int index, Pixel pixel, float mix)
{
    int x = 0, y = 0;
    line.position(index, x, y);
    const Rect window = context.output.bounds;
    if (x >= window.x1 && x < window.x2 && y >= window.y1 && y < window.y2) {
        context.output.pixel(x, y) = mix == 1.0f ? pixel : blend(context.source.read(x, y), pixel, mix);
    }
}

void finishRun(const Context& context, const Scanline& line, std::vector<SortValue>& run,
               int start, int minimumLength, bool descending, float mix)
{
    if (static_cast<int>(run.size()) >= minimumLength) {
        std::sort(run.begin(), run.end(), [descending](const SortValue& a, const SortValue& b) {
            if (a.key == b.key) {
                return a.index < b.index;
            }
            return descending ? a.key > b.key : a.key < b.key;
        });
    }
    for (const auto& value : run) {
        writeSortPixel(context, line, start++, value.pixel, mix);
    }
    run.clear();
}

struct PrintProfile {
    float contrast;
    float toe;
    float shoulder;
    float blackLift;
    float saturation;
    float warmHighlights;
    float coolShadows;
    float grain;
    bool monochrome;
};

constexpr PrintProfile profiles[] = {
    {1.00f, 0.10f, 0.32f, 0.005f, 1.00f,  0.00f, 0.00f, 0.90f, false},
    {1.06f, 0.14f, 0.43f, 0.010f, 0.98f,  0.28f, 0.08f, 1.00f, false},
    {1.10f, 0.17f, 0.42f, 0.008f, 0.87f, -0.26f, 0.32f, 0.98f, false},
    {1.15f, 0.19f, 0.40f, 0.006f, 0.00f,  0.00f, 0.00f, 1.20f, true},
};

float printTone(float value, const PrintProfile& profile, float contrast)
{
    constexpr float pivot = 0.42f;
    if (value < 0.0f) {
        return value * (1.0f - profile.toe) / contrast;
    }
    if (value < pivot) {
        const float ratio = value / pivot;
        const float shadow = 1.0f - ratio;
        return value * (1.0f - profile.toe * shadow * shadow) / (contrast + (1.0f - contrast) * ratio);
    }

    const float signal = pivot + contrast * (value - pivot);
    const float excess = std::max(0.0f, signal - 0.62f);
    // A smooth shoulder retains a positive slope for highlights above the nominal white point.
    return signal - profile.shoulder * excess * (excess / (excess + 0.24f));
}

std::uint32_t grainFrame(double time)
{
    constexpr double period = 4294967296.0;
    if (!std::isfinite(time)) {
        return 0;
    }
    const double wrapped = std::fmod(std::floor(time + 1.0e-6), period);
    return static_cast<std::uint32_t>(wrapped < 0.0 ? wrapped + period : wrapped);
}

float filmGrain(std::uint32_t seed, int x, int y, double sizeX, double sizeY)
{
    const float fine = 2.0f * randomAt(seed, x, y) - 1.0f;
    if (sizeX <= 1.0 && sizeY <= 1.0) {
        return fine;
    }

    const double gridX = (x + 0.5) / std::max(0.65, sizeX) + 0.37;
    const double gridY = (y + 0.5) / std::max(0.65, sizeY) + 0.21;
    const auto left = static_cast<int>(std::floor(gridX));
    const auto bottom = static_cast<int>(std::floor(gridY));
    const float u = smoothUnit(static_cast<float>(gridX - left));
    const float v = smoothUnit(static_cast<float>(gridY - bottom));
    const auto coarseSeed = seed ^ 0x94d049bbu;
    const float first = lerp(randomAt(coarseSeed, left, bottom), randomAt(coarseSeed, left + 1, bottom), u);
    const float second = lerp(randomAt(coarseSeed, left, bottom + 1), randomAt(coarseSeed, left + 1, bottom + 1), u);
    const float coarse = 2.0f * lerp(first, second, v) - 1.0f;
    const float coarseWeight = 0.75f * smoothUnit(static_cast<float>((std::max(sizeX, sizeY) - 1.0) / 0.65));
    return lerp(fine, coarse, coarseWeight);
}

float highlightCoverage(float luminance)
{
    constexpr float threshold = 0.68f;
    constexpr float knee = 0.12f;
    const float above = luminance - threshold;
    const float soft = std::clamp(above + knee, 0.0f, 2.0f * knee);
    return std::clamp(std::max(above, soft * soft / (4.0f * knee)) / std::max(luminance, 1.0e-6f), 0.0f, 1.0f);
}

}  // namespace

void renderPixelSort(const Context& context, const PixelSortSettings& settings)
{
    const float mix = static_cast<float>(std::clamp(settings.mix, 0.0, 1.0));
    const int frameWidth = width(context.frame);
    const int frameHeight = height(context.frame);
    if (mix == 0.0f || frameWidth == 0 || frameHeight == 0) {
        copySource(context);
        return;
    }

    constexpr double radians = 0.017453292519943295;
    const double angle = std::isfinite(settings.angle) ? std::fmod(settings.angle, 360.0) * radians : 0.0;
    const double par = context.pixelAspect > 0.0 ? context.pixelAspect : 1.0;
    const double xDirection = std::cos(angle) / par;
    const double yDirection = std::sin(angle);
    const bool horizontalMajor = std::abs(xDirection) >= std::abs(yDirection);
    const double majorDirection = horizontalMajor ? xDirection : yDirection;
    const double slope = (horizontalMajor ? yDirection : xDirection) / majorDirection;
    const int slopeSign = slope < 0.0 ? -1 : 1;
    const int majorLength = horizontalMajor ? frameWidth : frameHeight;
    const int minorLength = horizontalMajor ? frameHeight : frameWidth;
    const bool descending = settings.descending != (majorDirection < 0.0);
    const float low = static_cast<float>(std::min(settings.low, settings.high));
    const float high = static_cast<float>(std::max(settings.low, settings.high));

    // Rounding the same monotone shear for every line partitions the raster without gaps or overlaps.
    std::vector<int> offsets(static_cast<std::size_t>(majorLength));
    for (int index = 0; index < majorLength; ++index) {
        offsets[static_cast<std::size_t>(index)] = static_cast<int>(std::round(std::abs(slope) * index));
    }
    const int firstLine = slopeSign > 0 ? -offsets.back() : 0;
    const int lastLine = minorLength - 1 + (slopeSign < 0 ? offsets.back() : 0);
    const double fragments = 80.0 * std::pow(std::clamp(settings.fragmentation, 0.0, 1.0), 1.35);
    const auto seed = hash(static_cast<std::uint32_t>(settings.seed) ^ 0x4f1bbcdcu);
    // Each sheared line owns its pixels, so bands of lines can be sorted at the same time.
    parallelBands(firstLine, lastLine + 1, [&](int firstBandLine, int lastBandLine) {
        std::vector<SortValue> run;
        run.reserve(static_cast<std::size_t>(majorLength));
        for (int intercept = firstBandLine; intercept < lastBandLine; ++intercept) {
            if (context.cancelled()) {
                return;
            }
            const int minOffset = slopeSign > 0 ? -intercept : intercept - minorLength + 1;
            const int maxOffset = slopeSign > 0 ? minorLength - intercept : intercept + 1;
            const auto begin = std::lower_bound(offsets.begin(), offsets.end(), minOffset);
            const auto end = std::lower_bound(begin, offsets.end(), maxOffset);
            const int first = static_cast<int>(begin - offsets.begin());
            const int count = static_cast<int>(end - begin);
            if (count == 0) {
                continue;
            }

            const int minimum = std::max(2, static_cast<int>(std::ceil(count * settings.minLength / 100.0)));
            const int maximum = settings.maxLength <= 0.0 ? count
                : std::max(1, static_cast<int>(std::floor(count * settings.maxLength / 100.0)));
            const float breakChance = static_cast<float>(std::min(0.95, fragments / count));
            const Scanline line{horizontalMajor, intercept, slopeSign, offsets, context.frame};
            int runStart = first;
            for (int index = first; index < first + count; ++index) {
                int x = 0, y = 0;
                line.position(index, x, y);
                const Pixel pixel = context.source.read(x, y);
                const float luminance = pixel.a > 0.0f ? luma(pixel.r, pixel.g, pixel.b) / pixel.a : 0.0f;
                const bool inside = luminance >= low && luminance <= high;
                const bool eligible = pixel.a > 0.0f && std::isfinite(luminance) && (inside != settings.outside);
                const float key = eligible ? sortKey(settings.key, pixel, luminance) : 0.0f;
                const bool sortable = eligible && std::isfinite(key);
                const bool fragment = sortable && breakChance > 0.0f
                    && randomAt(seed, index, intercept) < breakChance;
                if (!run.empty() && (!sortable || static_cast<int>(run.size()) >= maximum || fragment)) {
                    finishRun(context, line, run, runStart, minimum, descending, mix);
                }
                if (sortable) {
                    if (run.empty()) {
                        runStart = index;
                    }
                    run.push_back({pixel, key, index});
                } else {
                    writeSortPixel(context, line, index, pixel, 1.0f);
                }
            }
            finishRun(context, line, run, runStart, minimum, descending, mix);
        }
    });
}

void renderFilmEffect(const Context& context, const FilmSettings& settings)
{
    const Rect window = context.output.bounds;
    const Rect frame = context.frame;
    const int frameWidth = width(frame);
    const int frameHeight = height(frame);
    const float mix = static_cast<float>(std::clamp(settings.mix, 0.0, 1.0));
    if (mix == 0.0f || frameWidth == 0 || frameHeight == 0) {
        copySource(context);
        return;
    }

    const auto& profile = profiles[std::clamp(settings.profile, 0, 3)];
    const float exposure = static_cast<float>(std::exp2(settings.exposure));
    const float contrast = static_cast<float>(settings.contrast) * profile.contrast;
    const float saturation = static_cast<float>(settings.saturation) * profile.saturation;
    const float fade = static_cast<float>(settings.fade) * 0.18f + profile.blackLift;
    const float warmth = static_cast<float>(settings.warmth) + profile.warmHighlights;
    const double sizeY = settings.grainSize * frameHeight / 1080.0;
    const double sizeX = sizeY / (context.pixelAspect > 0.0 ? context.pixelAspect : 1.0);
    const auto grainCoverage = static_cast<float>(std::min(1.0, std::sqrt(sizeX * sizeY)));
    const float grain = static_cast<float>(settings.grain) * profile.grain * (0.45f + 0.55f * grainCoverage);
    const auto seed = hash(hash(grainFrame(context.timeFrames)) ^ static_cast<std::uint32_t>(settings.seed) ^ 0x9e3779b9u);
    const float vignette = static_cast<float>(settings.vignette) * 0.82f;
    const float halation = static_cast<float>(settings.halation) * 0.72f;

    if (halation > 0.0f) {
        const GlowSettings highlightMask{0.68 / exposure, 0.12 / exposure, settings.halationRadius,
            1.0, 0.22, 1.0, 1.0, 1.0, 0, true};
        renderGlow(context, highlightMask);
    }

    if (context.cancelled()) {
        return;
    }
    forEachRow(context, window.y1, window.y2, [&](int y) {
        const int localY = y - frame.y1;
        const float v = static_cast<float>(2.0 * (localY + 0.5) / frameHeight - 1.0);
        for (int x = window.x1; x < window.x2; ++x) {
            const Pixel source = context.source.read(x, y);
            if (source.a <= 0.0f) {
                context.output.pixel(x, y) = source;
                continue;
            }

            const float exposedRed = source.r / source.a * exposure;
            const float exposedGreen = source.g / source.a * exposure;
            const float exposedBlue = source.b / source.a * exposure;
            const float luminance = luma(exposedRed, exposedGreen, exposedBlue);
            const float tone = printTone(luminance, profile, contrast);
            const float shadow = 1.0f - smoothUnit(tone / 0.54f);
            const float lifted = tone + fade * shadow * shadow;
            const float chroma = saturation * (std::abs(luminance) > 1.0e-5f
                ? std::clamp(tone / luminance, 0.0f, 4.0f) : 1.0f);

            float red = lifted + (exposedRed - luminance) * chroma;
            float green = lifted + (exposedGreen - luminance) * chroma;
            float blue = lifted + (exposedBlue - luminance) * chroma;
            if (!profile.monochrome) {
                const float highlights = smoothUnit((tone - 0.25f) / 0.55f);
                const float temperature = (warmth * highlights - profile.coolShadows * shadow) * std::max(0.0f, lifted);
                red += 0.12f * temperature;
                green -= 0.0195f * temperature;
                blue -= 0.16f * temperature;
            }

            if (halation > 0.0f) {
                // Subtract the direct highlight so warm scatter is strongest around bright edges.
                const float direct = source.a * highlightCoverage(luminance);
                const float scatter = std::max(0.0f, context.output.pixel(x, y).a - direct) * halation;
                red += scatter * (profile.monochrome ? 0.52f : 1.0f);
                green += scatter * (profile.monochrome ? 0.52f : 0.23f);
                blue += scatter * (profile.monochrome ? 0.52f : 0.055f);
            }

            const int localX = x - frame.x1;
            const float u = static_cast<float>(2.0 * (localX + 0.5) / frameWidth - 1.0);
            const float edge = smoothUnit((u * u + v * v - 0.30f) / 1.70f);
            const float falloff = 1.0f - vignette * edge;
            float speckle = 0.0f;
            if (grain > 0.0f) {
                const float midtone = 1.0f - std::abs(2.0f * std::clamp(lifted, 0.0f, 1.0f) - 1.0f);
                speckle = filmGrain(seed, localX, localY, sizeX, sizeY) * grain * (0.028f + 0.095f * midtone);
            }
            const Pixel printed{(red * falloff + speckle) * source.a,
                (green * falloff + speckle) * source.a, (blue * falloff + speckle) * source.a, source.a};
            context.output.pixel(x, y) = mix == 1.0f ? printed : blend(source, printed, mix);
        }
    });
}

}  // namespace diamond::fx
