#include "Video05.h"

#include <array>
#include <cstdint>
#include <vector>

namespace diamond::fx {
namespace {

constexpr double kPi = 3.14159265358979323846;

double bounded(double value, double low, double high, double fallback = 0) {
    return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}

float unit(double value) {
    return static_cast<float>(bounded(value, 0.0, 1.0));
}

double aspect(const Context& context) {
    return bounded(context.pixelAspect, 0.01, 100.0, 1.0);
}

float smooth(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

float luma(Pixel pixel) {
    return 0.2126f * pixel.r + 0.7152f * pixel.g + 0.0722f * pixel.b;
}

Pixel mix(Pixel source, Pixel effect, float weight) {
    return {source.r + (effect.r - source.r) * weight, source.g + (effect.g - source.g) * weight,
            source.b + (effect.b - source.b) * weight, source.a + (effect.a - source.a) * weight};
}

template <typename Render>
void renderPixels(const Context& context, const Render& render) {
    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            context.output.pixel(x, y) = render(x, y, context.source.read(x, y));
        }
    });
}

void copySource(const Context& context) {
    renderPixels(context, [](int, int, Pixel source) { return source; });
}

struct WidePixel {
    double r = 0, g = 0, b = 0, a = 0;

    WidePixel& operator+=(WidePixel other) {
        r += other.r;
        g += other.g;
        b += other.b;
        a += other.a;
        return *this;
    }
    WidePixel& operator-=(WidePixel other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
        a -= other.a;
        return *this;
    }
    WidePixel operator*(double weight) const { return {r * weight, g * weight, b * weight, a * weight}; }
};

double widen(float value) { return value; }
WidePixel widen(Pixel value) { return {value.r, value.g, value.b, value.a}; }
float narrow(double value) { return static_cast<float>(value); }
Pixel narrow(WidePixel value) {
    return {narrow(value.r), narrow(value.g), narrow(value.b), narrow(value.a)};
}

struct FractionalBox {
    int radius;
    int support;
    double wingWeight;
    double normalize;
};

FractionalBox gaussianBox(double sigma) {
    // Three equal boxes approximate a Gaussian. Choose fractional weights at
    // the two outermost taps to match the requested variance continuously.
    const double variance = sigma * sigma / 3.0;
    const int radius = static_cast<int>(std::floor((std::sqrt(1 + 12 * variance) - 1) * 0.5));
    const double taps = 2.0 * radius + 1;
    const double integerVariance = radius * (radius + 1.0) / 3;
    const double outer = radius + 1.0;
    const double wing = std::clamp(taps * (variance - integerVariance) /
        (2.0 * (outer * outer - variance)), 0.0, 1.0);
    return {radius, radius + (wing > 0.0 ? 1 : 0), wing, 1.0 / (taps + 2.0 * wing)};
}

bool mapCoordinate(std::int64_t& coordinate, int low, int high, Edge edge) {
    if (coordinate >= low && coordinate < high) {
        return true;
    }
    if (edge == Edge::Transparent) {
        return false;
    }
    const auto end = static_cast<std::int64_t>(high) - 1;
    const auto span = end - low;
    if (edge == Edge::Clamp || span == 0) {
        coordinate = std::clamp(coordinate, static_cast<std::int64_t>(low), end);
    } else {
        const auto period = 2 * span;
        auto offset = (coordinate - low) % period;
        if (offset < 0) {
            offset += period;
        }
        coordinate = low + std::min(offset, period - offset);
    }
    return true;
}

template <typename Value>
class LineFilter {
public:
    LineFilter(std::size_t size, FractionalBox box)
        : size_(size), box_(box), pad_(3 * static_cast<std::size_t>(box.support)),
          line_(size + 2 * pad_), scratch_(size + 2 * pad_) {}

    template <typename Load, typename Store>
    void run(int begin, const Load& load, const Store& store) {
        const auto first = static_cast<std::int64_t>(begin) - static_cast<std::int64_t>(pad_);
        for (std::size_t i = 0; i < line_.size(); ++i) {
            line_[i] = load(first + static_cast<std::int64_t>(i));
        }
        std::size_t active = line_.size();
        for (int pass = 0; pass < 3 && box_.support != 0; ++pass) {
            active = boxPass(active);
            line_.swap(scratch_);
        }
        for (std::size_t i = 0; i < size_; ++i) {
            store(begin + static_cast<int>(i), line_[i]);
        }
    }

private:
    std::size_t boxPass(std::size_t size) {
        using Accumulator = decltype(widen(Value{}));
        const auto inner = static_cast<std::size_t>(box_.radius);
        const auto outer = static_cast<std::size_t>(box_.support);
        const std::size_t valid = size - 2 * outer;
        const std::size_t first = outer - inner;
        const std::size_t last = outer + inner;
        Accumulator sum{};
        for (std::size_t i = first; i <= last; ++i) {
            sum += widen(line_[i]);
        }
        for (std::size_t i = 0; i < valid; ++i) {
            auto weighted = sum;
            if (box_.wingWeight > 0.0) {
                weighted += widen(line_[i]) * box_.wingWeight;
                weighted += widen(line_[i + 2 * outer]) * box_.wingWeight;
            }
            scratch_[i] = narrow(weighted * box_.normalize);
            if (i + 1 < valid) {
                sum -= widen(line_[i + first]);
                sum += widen(line_[i + last + 1]);
            }
        }
        return valid;
    }

    std::size_t size_;
    FractionalBox box_;
    std::size_t pad_;
    std::vector<Value> line_;
    std::vector<Value> scratch_;
};

template <typename Value, typename Read, typename Write>
void gaussian(const Context& context, double radius, Edge edge, const Read& read, const Write& write) {
    const Rect source = context.source.bounds;
    const Rect window = context.output.bounds;
    const int columns = width(window);
    const int sourceRows = height(source);
    if (columns == 0 || height(window) == 0) {
        return;
    }
    if (width(source) == 0 || sourceRows == 0) {
        renderPixels(context, [&](int x, int y, Pixel) { return write(x, y, Value{}); });
        return;
    }

    const double sigma = 0.5 * radius;
    const FractionalBox horizontalBox = gaussianBox(sigma / aspect(context));
    const FractionalBox verticalBox = gaussianBox(sigma);
    std::vector<Value> intermediate(static_cast<std::size_t>(columns) * sourceRows);
    const auto index = [&](int x, int y) {
        return static_cast<std::size_t>(y - source.y1) * columns + (x - window.x1);
    };
    // A line filter holds scratch rows, so every band of lines gets its own.
    parallelBands(source.y1, source.y2, [&](int firstRow, int lastRow) {
        LineFilter<Value> horizontal(columns, horizontalBox);
        for (int y = firstRow; y < lastRow; ++y) {
            if (context.cancelled()) {
                return;
            }
            horizontal.run(window.x1, [&](std::int64_t x) {
                return mapCoordinate(x, source.x1, source.x2, edge) ? read(static_cast<int>(x), y) : Value{};
            }, [&](int x, Value value) {
                intermediate[index(x, y)] = value;
            });
        }
    });
    if (context.cancelled()) {
        return;
    }
    parallelBands(window.x1, window.x2, [&](int firstColumn, int lastColumn) {
        LineFilter<Value> vertical(height(window), verticalBox);
        for (int x = firstColumn; x < lastColumn; ++x) {
            if (context.cancelled()) {
                return;
            }
            vertical.run(window.y1, [&](std::int64_t y) {
                return mapCoordinate(y, source.y1, source.y2, edge)
                    ? intermediate[index(x, static_cast<int>(y))] : Value{};
            }, [&](int y, Value value) {
                context.output.pixel(x, y) = write(x, y, value);
            });
        }
    });
}

std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float noise(int x, int y, std::uint32_t seed) {
    const auto key = static_cast<std::uint32_t>(x) * 0x9e3779b9U
        ^ static_cast<std::uint32_t>(y) * 0x85ebca6bU ^ seed;
    return static_cast<float>(hash(key) >> 8) / 8388608.0f - 1.0f;
}

float grainField(double x, double y, std::uint32_t seed) {
    const int left = static_cast<int>(std::floor(x));
    const int bottom = static_cast<int>(std::floor(y));
    const float u = smooth(static_cast<float>(x - left));
    const float v = smooth(static_cast<float>(y - bottom));
    const float a = noise(left, bottom, seed);
    const float b = noise(left + 1, bottom, seed);
    const float c = noise(left, bottom + 1, seed);
    const float d = noise(left + 1, bottom + 1, seed);
    const float low = a + (b - a) * u;
    const float high = c + (d - c) * u;
    return low + (high - low) * v;
}

std::uint32_t grainFrame(const Context& context, double speed) {
    constexpr double period = 4294967296.0;
    const double frame = context.timeFrames * speed / (context.fps > 0.0 ? context.fps : 30.0);
    if (!std::isfinite(frame)) {
        return 0;
    }
    const double index = std::fmod(std::floor(frame), period);
    return static_cast<std::uint32_t>(index < 0.0 ? index + period : index);
}

float highlight(Pixel input, float threshold, float knee) {
    const float brightness = luma(input);
    if (!std::isfinite(brightness)) {
        return 0.0f;
    }
    const float signal = brightness - threshold;
    if (knee <= 0.0f) {
        return std::max(0.0f, signal);
    }
    const float weight = std::clamp(signal + knee, 0.0f, 2.0f * knee) / (2.0f * knee);
    return std::max(0.0f, signal) * smooth(weight);
}

} // namespace

void renderBlur(const Context& context, const BlurSettings05& settings) {
    const auto weight = unit(settings.mix);
    const double radius = height(context.frame) * 0.01 * bounded(settings.radius, 0, 15);
    if (radius <= 0 || weight <= 0) {
        copySource(context);
        return;
    }
    gaussian<Pixel>(context, radius, settings.edge,
        [&](int x, int y) { return context.source.read(x, y); },
        [&](int x, int y, Pixel blurred) {
            return weight == 1 ? blurred : mix(context.source.read(x, y), blurred, weight);
        });
}

void renderBlurDirectional(const Context& context, const DirectionalBlurSettings& settings) {
    const auto weight = unit(settings.mix);
    const double length = height(context.frame) * 0.01 * bounded(settings.length, 0, 35);
    if (length <= 0 || weight <= 0) {
        copySource(context);
        return;
    }
    const double angle = bounded(settings.angle, -360, 360) * (kPi / 180.0);
    const double dx = std::cos(angle) * length / aspect(context);
    const double dy = std::sin(angle) * length;
    const double bias = bounded(settings.bias, -1, 1) * 0.5;
    struct Tap { double x, y; float weight; };
    std::array<Tap, 96> taps{};
    const int count = std::clamp(settings.samples, 4, static_cast<int>(taps.size()));
    float total = 0;
    for (int i = 0; i < count; ++i) {
        const double position = (i + 0.5) / count - 0.5;
        const float envelope = static_cast<float>(0.65 + 0.35 * std::cos(2 * kPi * position));
        taps[i] = {-(position + bias) * dx, -(position + bias) * dy, envelope};
        total += envelope;
    }
    for (int i = 0; i < count; ++i) {
        taps[i].weight /= total;
    }
    renderPixels(context, [&](int x, int y, Pixel source) {
        Pixel blurred{};
        for (int i = 0; i < count; ++i) {
            const Tap& tap = taps[i];
            const Pixel value = context.source.sample(x + tap.x, y + tap.y, settings.edge);
            blurred.r += tap.weight * value.r;
            blurred.g += tap.weight * value.g;
            blurred.b += tap.weight * value.b;
            blurred.a += tap.weight * value.a;
        }
        return weight == 1 ? blurred : mix(source, blurred, weight);
    });
}

void renderSharpen(const Context& context, const SharpenSettings& settings) {
    const auto strength = static_cast<float>(bounded(settings.amount, 0, 4)) * unit(settings.mix);
    const double radius = height(context.frame) * 0.01 * bounded(settings.radius, 0, 4);
    if (radius <= 0 || strength <= 0) {
        copySource(context);
        return;
    }
    const float threshold = unit(settings.threshold);
    gaussian<Pixel>(context, radius, Edge::Reflect,
        [&](int x, int y) { return context.source.read(x, y); },
        [&](int x, int y, Pixel blurred) {
            const Pixel source = context.source.read(x, y);
            if (source.a <= 1.0e-6f || blurred.a <= 1.0e-6f) {
                return source;
            }
            const float ratio = source.a / blurred.a;
            const float dr = source.r - blurred.r * ratio;
            const float dg = source.g - blurred.g * ratio;
            const float db = source.b - blurred.b * ratio;
            const float detail = std::max({std::abs(dr), std::abs(dg), std::abs(db)}) / source.a;
            if (detail <= 0) {
                return source;
            }
            // A soft limiter restrains extreme halo contrast while preserving
            // the original high-dynamic-range values underneath the detail.
            const float gate = threshold > 0 ? smooth((detail - threshold) / threshold) : 1;
            const float gain = strength * gate / (1 + 0.8f * detail);
            return Pixel{source.r + dr * gain, source.g + dg * gain, source.b + db * gain, source.a};
        });
}

void renderVignette(const Context& context, const VignetteSettings& settings) {
    const float amount = unit(settings.amount);
    if (amount <= 0) {
        copySource(context);
        return;
    }
    const double par = aspect(context);
    const double frameWidth = std::max(1, width(context.frame));
    const double frameHeight = std::max(1, height(context.frame));
    const double centerX = context.frame.x1 - 0.5 + frameWidth * bounded(settings.centerX, -1, 2, 0.5);
    const double centerY = context.frame.y1 - 0.5 + frameHeight * bounded(settings.centerY, -1, 2, 0.5);
    const double angle = bounded(settings.angle, -360, 360) * (kPi / 180.0);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double radius = bounded(settings.radius, 0.02, 2, 0.64);
    const double softness = unit(settings.softness);
    const double feather = std::max(2.0 / frameHeight, 0.72 * radius * softness);
    const double inner = std::max(0.0, radius - feather);
    const double outer = radius + feather;
    const double exponent = 2.0 + (1.0 - unit(settings.roundness)) * 8.0;
    const float red = unit(settings.red);
    const float green = unit(settings.green);
    const float blue = unit(settings.blue);
    renderPixels(context, [&](int x, int y, Pixel source) {
        const double dx = (x - centerX) * par;
        const double dy = y - centerY;
        const double u = std::abs((cosine * dx + sine * dy) / (0.5 * frameWidth * par));
        const double v = std::abs((cosine * dy - sine * dx) / (0.5 * frameHeight));
        const double distance = std::pow(std::pow(u, exponent) + std::pow(v, exponent), 1.0 / exponent);
        const float weight = amount * smooth(static_cast<float>((distance - inner) / (outer - inner)));
        return Pixel{source.r + (source.a * red - source.r) * weight,
                     source.g + (source.a * green - source.g) * weight,
                     source.b + (source.a * blue - source.b) * weight, source.a};
    });
}

void renderGrain(const Context& context, const GrainSettings& settings) {
    const float amount = unit(settings.amount);
    if (amount <= 0 || height(context.frame) == 0) {
        copySource(context);
        return;
    }
    const double grainSize = bounded(settings.size, 0.25, 12, 1.35)
        * height(context.frame) / 1080.0;
    const double par = aspect(context);
    const auto footprint = static_cast<float>(grainSize / std::max(par, 1.0));
    const float fine = unit(settings.roughness) * smooth((footprint - 0.42f) * 0.9f);
    const float coarse = smooth(footprint * 1.35f);
    const float color = unit(settings.color);
    const float response = unit(settings.response);
    const double speed = bounded(settings.speed, 0, 120);
    const auto seed = hash(static_cast<std::uint32_t>(settings.seed) ^ hash(grainFrame(context, speed)));
    renderPixels(context, [&](int x, int y, Pixel source) {
        if (source.a <= 1.0e-6f) {
            return source;
        }
        const double gx = ((x - context.frame.x1) + 0.5) * par / grainSize;
        const double gy = ((y - context.frame.y1) + 0.5) / grainSize;
        const float luminance = std::max(0.0f, luma(source) / source.a);
        const float tone = std::min(luminance, 1.0f);
        const float midtone = 4.0f * tone * (1.0f - tone);
        const float gain = 0.25f * source.a * amount * (1.0f - response * (0.88f - 0.68f * midtone));
        const float mono = grainField(gx, gy, seed) * coarse
            + grainField(gx * 2.13 + 12.0, gy * 2.13 - 7.0, seed ^ 0xb8a37413U) * (0.65f * fine);
        const float base = mono * gain * (1.0f - 0.36f * color);
        const float chroma = gain * coarse * color * 0.58f;
        const auto channel = [&, gx, gy](float value, std::uint32_t salt) {
            const float variation = chroma > 0 ? chroma * grainField(gx, gy, seed ^ salt) : 0;
            // Keep nonnegative input channels nonnegative, while leaving
            // pre-existing negative scene-referred values intact.
            return value + std::max(base + variation, -std::max(0.0f, value));
        };
        return Pixel{channel(source.r, 0x2cd39a37U), channel(source.g, 0x8b7352c7U),
                     channel(source.b, 0xab60da19U), source.a};
    });
}

void renderHalation(const Context& context, const HalationSettings& settings) {
    const auto amount = static_cast<float>(bounded(settings.amount, 0, 4));
    const double radius = height(context.frame) * 0.01 * bounded(settings.radius, 0, 8);
    if (radius <= 0 || amount <= 0) {
        if (settings.halationOnly) {
            renderPixels(context, [](int, int, Pixel) { return Pixel{}; });
        } else {
            copySource(context);
        }
        return;
    }
    const auto threshold = static_cast<float>(bounded(settings.threshold, 0, 4, 0.62));
    const auto knee = static_cast<float>(bounded(settings.knee, 0, 2, 0.2));
    const float red = unit(settings.red);
    const float green = unit(settings.green);
    const float blue = unit(settings.blue);
    gaussian<float>(context, radius * 1.6, Edge::Transparent,
        [&](int x, int y) { return highlight(context.source.read(x, y), threshold, knee); },
        [&](int x, int y, float blurred) {
            const Pixel source = context.source.read(x, y);
            const float emitter = highlight(source, threshold, knee);
            const float spill = std::max(0.0f, blurred - 0.80f * emitter) * amount;
            const float r = spill * red;
            const float g = spill * green;
            const float b = spill * blue;
            const float coverage = std::clamp(std::max({r, g, b}), 0.0f, 1.0f);
            const Pixel base = settings.halationOnly ? Pixel{} : source;
            return Pixel{base.r + r, base.g + g, base.b + b, base.a + (1.0f - base.a) * coverage};
        });
}

} // namespace diamond::fx
