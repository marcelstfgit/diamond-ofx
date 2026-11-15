#include "Effects.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace diamond::fx {
namespace {

Pixel scaled(Pixel pixel, float scale)
{
    return {pixel.r * scale, pixel.g * scale, pixel.b * scale, pixel.a * scale};
}

void add(Pixel& sum, const Pixel& value)
{
    sum.r += value.r;
    sum.g += value.g;
    sum.b += value.b;
    sum.a += value.a;
}

void advance(Pixel& sum, const Pixel& incoming, const Pixel& outgoing)
{
    sum.r += incoming.r - outgoing.r;
    sum.g += incoming.g - outgoing.g;
    sum.b += incoming.b - outgoing.b;
    sum.a += incoming.a - outgoing.a;
}

Pixel mix(Pixel first, Pixel second, float amount)
{
    first = scaled(first, 1.0f - amount);
    add(first, scaled(second, amount));
    return first;
}

void copySource(const Context& context)
{
    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            context.output.pixel(x, y) = context.source.read(x, y);
        }
    });
}

std::size_t offset(int x, int y, int width)
{
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

float finitePositive(float value)
{
    return std::isfinite(value) ? std::max(0.0f, value) : 0.0f;
}

Pixel highlight(Pixel pixel, float threshold, float knee)
{
    const float alpha = std::clamp(finitePositive(pixel.a), 0.0f, 1.0f);
    if (alpha == 0.0f) {
        return {};
    }

    pixel.r = finitePositive(pixel.r);
    pixel.g = finitePositive(pixel.g);
    pixel.b = finitePositive(pixel.b);
    const double luma = (0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b) / alpha;
    if (luma <= 1.0e-6) {
        return {};
    }
    const double above = luma - threshold;
    const double kneeInput = std::clamp(above + knee, 0.0, 2.0 * knee);
    const double soft = knee > 0.0f ? kneeInput * (kneeInput / (4.0 * knee)) : 0.0;
    const float fraction = static_cast<float>(std::clamp(std::max(above, soft) / luma, 0.0, 1.0));
    pixel.a = alpha;
    return scaled(pixel, fraction);
}

bool blurRows(const Context& context, std::vector<Pixel>& pixels, int width, int height, int radius)
{
    if (radius == 0) {
        return true;
    }

    const float weight = 1.0f / (2.0f * static_cast<float>(radius) + 1.0f);
    parallelBands(0, height, [&](int firstRow, int lastRow) {
        std::vector<Pixel> line(static_cast<std::size_t>(width));
        for (int y = firstRow; y < lastRow; ++y) {
            if (context.cancelled()) {
                return;
            }
            auto* row = pixels.data() + offset(0, y, width);
            Pixel sum = scaled(row[0], static_cast<float>(radius + 1));
            for (int x = 1; x <= radius; ++x) {
                add(sum, row[std::min(x, width - 1)]);
            }
            for (int x = 0; x < width; ++x) {
                line[static_cast<std::size_t>(x)] = scaled(sum, weight);
                advance(sum, row[std::min(x + radius + 1, width - 1)], row[std::max(x - radius, 0)]);
            }
            std::copy_n(line.data(), width, row);
        }
    });
    return !context.cancelled();
}

bool blurColumns(const Context& context, std::vector<Pixel>& pixels, int width, int height, int radius)
{
    if (radius == 0) {
        return true;
    }

    // Keep several neighboring columns together so the vertical pass reads contiguous pixels.
    constexpr int columns = 32;
    const float weight = 1.0f / (2.0f * static_cast<float>(radius) + 1.0f);
    // A stripe reads and writes only its own columns, so stripes can run side by side.
    parallelBands(0, (width + columns - 1) / columns, [&](int firstStripe, int lastStripe) {
        std::vector<Pixel> stripe(offset(0, height, columns));
        for (int index = firstStripe; index < lastStripe; ++index) {
            if (context.cancelled()) {
                return;
            }
            const int left = index * columns;
            const int count = std::min(columns, width - left);
            Pixel sums[columns]{};
            for (int i = 0; i < count; ++i) {
                sums[i] = scaled(pixels[offset(left + i, 0, width)], static_cast<float>(radius + 1));
            }
            for (int y = 1; y <= radius; ++y) {
                const auto* row = pixels.data() + offset(left, std::min(y, height - 1), width);
                for (int i = 0; i < count; ++i) {
                    add(sums[i], row[i]);
                }
            }
            for (int y = 0; y < height; ++y) {
                const auto* incoming = pixels.data() + offset(left, std::min(y + radius + 1, height - 1), width);
                const auto* outgoing = pixels.data() + offset(left, std::max(y - radius, 0), width);
                auto* row = stripe.data() + offset(0, y, columns);
                for (int i = 0; i < count; ++i) {
                    row[i] = scaled(sums[i], weight);
                    advance(sums[i], incoming[i], outgoing[i]);
                }
            }
            for (int y = 0; y < height; ++y) {
                std::copy_n(stripe.data() + offset(0, y, columns), count, pixels.data() + offset(left, y, width));
            }
        }
    });
    return !context.cancelled();
}

bool soften(const Context& context, std::vector<Pixel>& light, int width, int height, int radiusX, int radiusY)
{
    // Two cascaded boxes produce a tapered falloff while keeping cost linear in the image size.
    return blurRows(context, light, width, height, (radiusX + 1) / 2)
        && blurColumns(context, light, width, height, (radiusY + 1) / 2)
        && blurRows(context, light, width, height, radiusX / 2)
        && blurColumns(context, light, width, height, radiusY / 2);
}

int blurRadius(double requested, int dimension)
{
    if (std::isnan(requested)) {
        return 0;
    }
    return static_cast<int>(std::round(std::clamp(requested, 0.0, static_cast<double>(dimension - 1))));
}

Pixel compositeGlow(Pixel source, Pixel glow, const Pixel& tint, bool screen, bool glowOnly)
{
    glow.r = finitePositive(glow.r) * tint.r;
    glow.g = finitePositive(glow.g) * tint.g;
    glow.b = finitePositive(glow.b) * tint.b;
    glow.a = std::clamp(finitePositive(glow.a) * tint.a, 0.0f, 1.0f);
    if (glowOnly) {
        return glow;
    }

    if (screen) {
        glow.r *= std::max(0.0f, 1.0f - source.r);
        glow.g *= std::max(0.0f, 1.0f - source.g);
        glow.b *= std::max(0.0f, 1.0f - source.b);
    }
    glow.a *= 1.0f - std::clamp(source.a, 0.0f, 1.0f);
    add(source, glow);
    return source;
}

std::uint32_t hash(std::uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float random(std::uint32_t key, int x, int y, std::uint32_t stream = 0)
{
    const auto bits = hash(key ^ (0x9e3779b9u * static_cast<std::uint32_t>(x))
        ^ (0x85ebca6bu * static_cast<std::uint32_t>(y)) ^ (0xc2b2ae35u * stream));
    return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

std::uint32_t damageTick(const Context& context, double frequency)
{
    const double fps = context.fps > 0.0 ? context.fps : 30.0;
    const double tick = std::floor(context.timeFrames * frequency / fps);
    if (!std::isfinite(tick)) {
        return 0;
    }
    constexpr double period = 4294967296.0;
    const double wrapped = std::fmod(tick, period);
    return static_cast<std::uint32_t>(wrapped < 0.0 ? wrapped + period : wrapped);
}

int wrapX(int x, int displacement, const Rect& frame)
{
    const auto width = static_cast<std::int64_t>(frame.x2) - frame.x1;
    const auto relative = static_cast<std::int64_t>(x) - frame.x1 + displacement;
    if (relative >= 0 && relative < width) {
        return static_cast<int>(frame.x1 + relative);
    }
    const auto wrapped = relative % width;
    return static_cast<int>(frame.x1 + (wrapped < 0 ? wrapped + width : wrapped));
}

struct Block {
    int offset = 0;
    float colorShift = 0.0f;
    float levels = 0.0f;
};

Block blockAt(std::uint32_t key, int column, int row, int size, double amount)
{
    if (amount == 0.0 || random(key, column, row) >= 0.48 * amount) {
        return {};
    }
    const float style = random(key, column, row, 1);
    if (style < 0.48f) {
        const auto distance = 1 + static_cast<int>(random(key, column, row, 2) * 4.0f);
        return {(style < 0.24f ? -distance : distance) * size, (style - 0.24f) * 0.35f, 12.0f};
    }
    return {0, (style - 0.72f) * 0.60f, 4.0f + 8.0f * random(key, column, row, 3)};
}

float splitChannel(const Context& context, int x, int y, int offsetX, float fraction,
                   float Pixel::* channel, float current, float alpha)
{
    const int left = std::clamp(x + offsetX, context.frame.x1, context.frame.x2 - 1);
    const int right = std::clamp(x + offsetX + 1, context.frame.x1, context.frame.x2 - 1);
    const Pixel p = context.source.read(left, y);
    const Pixel q = context.source.read(right, y);
    const float replacementAlpha = p.a + (q.a - p.a) * fraction;
    if (replacementAlpha <= 1.0e-5f) {
        return current;
    }
    const float replacement = p.*channel + (q.*channel - p.*channel) * fraction;
    return replacement * (alpha / replacementAlpha);
}

float quantize(float value, float levels, float alpha)
{
    return std::round(value * (levels / alpha)) * (alpha / levels);
}

}  // namespace

void renderGlow(const Context& context, const GlowSettings& settings)
{
    const Pixel tint = {static_cast<float>(settings.intensity * settings.red),
        static_cast<float>(settings.intensity * settings.green), static_cast<float>(settings.intensity * settings.blue),
        static_cast<float>(settings.intensity * std::max({settings.red, settings.green, settings.blue}))};
    if (tint.a == 0.0f) {
        if (!settings.glowOnly) {
            copySource(context);
        }
        return;
    }

    const Rect window = context.output.bounds;
    const Rect frame = context.frame;
    const int frameWidth = frame.x2 - frame.x1;
    const int frameHeight = frame.y2 - frame.y1;
    if (window.x1 >= window.x2 || window.y1 >= window.y2 || frameWidth <= 0 || frameHeight <= 0) {
        return;
    }

    const double requestedY = settings.radius * frameHeight / 100.0;
    const double requestedX = requestedY / (context.pixelAspect > 0.0 ? context.pixelAspect : 1.0);
    const int radiusX = blurRadius(requestedX, frameWidth);
    const int radiusY = blurRadius(requestedY, frameHeight);
    const float spread = static_cast<float>(std::clamp(settings.spread, 0.0, 1.0));
    const int extraX = spread > 0.0f ? blurRadius(requestedX * 2.0, frameWidth) : 0;
    const int extraY = spread > 0.0f ? blurRadius(requestedY * 2.0, frameHeight) : 0;
    const Rect area = {std::max(frame.x1, window.x1 - radiusX - extraX),
        std::max(frame.y1, window.y1 - radiusY - extraY),
        std::min(frame.x2, window.x2 + radiusX + extraX),
        std::min(frame.y2, window.y2 + radiusY + extraY)};
    const int width = area.x2 - area.x1;
    const int height = area.y2 - area.y1;
    if (width <= 0 || height <= 0) {
        return;
    }

    std::vector<Pixel> light(offset(0, height, width));
    const float threshold = static_cast<float>(settings.threshold);
    const float knee = static_cast<float>(std::max(0.0, settings.knee));
    forEachRow(context, area.y1, area.y2, [&](int y) {
        for (int x = area.x1; x < area.x2; ++x) {
            light[offset(x - area.x1, y - area.y1, width)] = highlight(context.source.read(x, y), threshold, knee);
        }
    });
    if (context.cancelled() || !soften(context, light, width, height, radiusX, radiusY)) {
        return;
    }

    const Rect active = {std::max(window.x1, frame.x1), std::max(window.y1, frame.y1),
        std::min(window.x2, frame.x2), std::min(window.y2, frame.y2)};
    const bool broad = extraX > 0 || extraY > 0;
    if (broad) {
        forEachRow(context, active.y1, active.y2, [&](int y) {
            for (int x = active.x1; x < active.x2; ++x) {
                context.output.pixel(x, y) = light[offset(x - area.x1, y - area.y1, width)];
            }
        });
        if (context.cancelled() || !soften(context, light, width, height, extraX, extraY)) {
            return;
        }
    }

    forEachRow(context, active.y1, active.y2, [&](int y) {
        for (int x = active.x1; x < active.x2; ++x) {
            Pixel glow = light[offset(x - area.x1, y - area.y1, width)];
            auto& target = context.output.pixel(x, y);
            if (broad) {
                glow = mix(target, glow, spread);
            }
            target = compositeGlow(context.source.read(x, y), glow, tint, settings.blend == 1, settings.glowOnly);
        }
    });
}

void renderDigitalDamage(const Context& context, const DamageSettings& settings)
{
    const Rect window = context.output.bounds;
    const Rect frame = context.frame;
    const int width = frame.x2 - frame.x1;
    const int height = frame.y2 - frame.y1;
    if (width <= 0 || height <= 0) {
        return;
    }

    const float amount = static_cast<float>(std::clamp(settings.amount, 0.0, 1.0));
    if (amount == 0.0f) {
        copySource(context);
        return;
    }
    const double displacement = settings.displacement * width / 100.0;
    const double chroma = settings.chroma * width / 100.0;
    const auto tick = damageTick(context, settings.frequency);
    const auto seed = static_cast<std::uint32_t>(settings.seed);
    const auto pattern = hash(seed ^ hash(tick) ^ 0xa511e9b3u);
    const auto blockPattern = hash(seed ^ hash(tick / 2u) ^ 0x94d049bbu);
    const int bandHeight = std::max(2, height / (24 + static_cast<int>(pattern % 11u)));
    const int blockSize = std::clamp(height / 37, 4, 160);
    const int dropoutHeight = std::max(1, height / 320);
    const int dropoutWidth = 3 * blockSize;
    const double scanPeriod = std::max(2.0, height / 260.0);
    const double scanPhase = random(pattern, 0, 0, 7) * scanPeriod;
    const float noiseScale = static_cast<float>(0.65 * settings.noise);

    forEachRow(context, window.y1, window.y2, [&](int y) {
        const int row = y - frame.y1;
        const int band = (row + bandHeight / 2) / bandHeight;
        const bool tearing = random(pattern, band, 0) < 0.36f;
        const double pull = 2.0 * random(pattern, band, 0, 1) - 1.0;
        const int bandOffset = tearing ? static_cast<int>(std::round(pull * displacement)) : 0;
        const double split = chroma * (tearing ? 0.65 + 0.35 * std::abs(pull) : 0.30);
        const int redOffset = static_cast<int>(std::floor(split));
        const int blueOffset = static_cast<int>(std::floor(-split));
        const float redFraction = static_cast<float>(split - redOffset);
        const float blueFraction = static_cast<float>(-split - blueOffset);
        constexpr double tau = 6.283185307179586;
        const double scan = settings.scanlines > 0.0 ? 0.5 + 0.5 * std::cos(tau * (row + scanPhase) / scanPeriod) : 0.0;
        const float scanGain = 1.0f - static_cast<float>(0.55 * settings.scanlines * scan * scan);

        int blockColumn = -1;
        int dropoutColumn = -1;
        Block block;
        float dropoutGain = 1.0f;
        for (int x = window.x1; x < window.x2; ++x) {
            const Pixel original = context.source.read(x, y);
            const int column = x - frame.x1;
            if (column / blockSize != blockColumn) {
                blockColumn = column / blockSize;
                block = blockAt(blockPattern, blockColumn, row / blockSize, blockSize, settings.blocks);
            }
            if (column / dropoutWidth != dropoutColumn) {
                dropoutColumn = column / dropoutWidth;
                const float roll = random(pattern, dropoutColumn, row / dropoutHeight, 4);
                dropoutGain = roll < 0.18 * settings.dropout ? (roll < 0.025 * settings.dropout ? 1.45f : 0.18f) : 1.0f;
            }

            const int shift = bandOffset + block.offset;
            const int sampleX = shift != 0 ? wrapX(x, shift, frame) : x;
            Pixel damaged = shift != 0 ? context.source.read(sampleX, y) : original;
            if (split > 0.0 && damaged.a > 0.0f) {
                damaged.r = splitChannel(context, sampleX, y, redOffset, redFraction, &Pixel::r, damaged.r, damaged.a);
                damaged.b = splitChannel(context, sampleX, y, blueOffset, blueFraction, &Pixel::b, damaged.b, damaged.a);
            }
            if (block.levels > 0.0f && damaged.a > 1.0e-5f) {
                damaged.r = quantize(damaged.r, block.levels, damaged.a) + block.colorShift * damaged.a;
                damaged.g = quantize(damaged.g, block.levels, damaged.a);
                damaged.b = quantize(damaged.b, block.levels, damaged.a) - block.colorShift * damaged.a;
            }
            const float noise = noiseScale > 0.0f
                ? (random(pattern, column, row, 9) - 0.5f) * noiseScale * damaged.a : 0.0f;
            const float gain = scanGain * dropoutGain;
            damaged.r = damaged.r * gain + noise;
            damaged.g = damaged.g * gain + noise;
            damaged.b = damaged.b * gain + noise;
            context.output.pixel(x, y) = mix(original, damaged, amount);
        }
    });
}

}  // namespace diamond::fx
