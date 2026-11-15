#pragma once

#include "Parallel.h"
#include "Shake.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace diamond::fx {

struct Pixel {
    float r = 0, g = 0, b = 0, a = 0;
};

inline int width(Rect bounds) { return std::max(0, bounds.x2 - bounds.x1); }
inline int height(Rect bounds) { return std::max(0, bounds.y2 - bounds.y1); }

class Image {
public:
    Rect bounds{};

    Image() = default;
    explicit Image(Rect area) : bounds(area), stride_(width(area)),
        pixels_(static_cast<std::size_t>(stride_) * height(area)) {}

    Pixel& pixel(int x, int y) { return pixels_[index(x, y)]; }
    Pixel* data() { return pixels_.data(); }
    const Pixel* data() const { return pixels_.data(); }

    Pixel read(int x, int y) const
    {
        if (x < bounds.x1 || x >= bounds.x2 || y < bounds.y1 || y >= bounds.y2) {
            return {};
        }
        return pixels_[index(x, y)];
    }

    Pixel sample(double x, double y, Edge edge = Edge::Transparent) const
    {
        if (pixels_.empty()) {
            return {};
        }
        x = extend(x, bounds.x1, bounds.x2, edge);
        y = extend(y, bounds.y1, bounds.y2, edge);
        if (!(x > bounds.x1 - 1.0 && x < bounds.x2
           && y > bounds.y1 - 1.0 && y < bounds.y2)) {
            return {};
        }

        const int left = static_cast<int>(std::floor(x));
        const int bottom = static_cast<int>(std::floor(y));
        const auto u = static_cast<float>(x - left);
        const auto v = static_cast<float>(y - bottom);
        if (left >= bounds.x1 && left < bounds.x2 - 1
            && bottom >= bounds.y1 && bottom < bounds.y2 - 1) {
            const auto i = index(left, bottom);
            return bilinear(pixels_[i], pixels_[i + 1], pixels_[i + stride_],
                            pixels_[i + stride_ + 1], u, v);
        }
        return bilinear(read(left, bottom), read(left + 1, bottom),
                        read(left, bottom + 1), read(left + 1, bottom + 1), u, v);
    }

private:
    int stride_ = 0;
    std::vector<Pixel> pixels_;

    std::size_t index(int x, int y) const
    {
        return static_cast<std::size_t>(y - bounds.y1) * stride_ + (x - bounds.x1);
    }

    static double extend(double position, int low, int high, Edge edge)
    {
        const double span = static_cast<double>(high) - low - 1.0;
        const double relative = position - low;
        if (edge == Edge::Transparent || (relative >= 0 && relative <= span)) {
            return position;
        }
        if (edge == Edge::Clamp) {
            return std::clamp(position, static_cast<double>(low), high - 1.0);
        }
        if (span == 0) {
            return low;
        }
        const double period = 2.0 * span;
        double offset = std::fmod(relative, period);
        if (offset < 0) {
            offset += period;
        }
        return low + std::min(offset, period - offset);
    }

    static Pixel bilinear(Pixel p, Pixel q, Pixel r, Pixel s, float u, float v)
    {
        const float a = (1 - u) * (1 - v);
        const float b = u * (1 - v);
        const float c = (1 - u) * v;
        const float d = u * v;
        return {
            p.r * a + q.r * b + r.r * c + s.r * d,
            p.g * a + q.g * b + r.g * c + s.g * d,
            p.b * a + q.b * b + r.b * c + s.b * d,
            p.a * a + q.a * b + r.a * c + s.a * d
        };
    }
};

struct Context {
    const Image& source;
    Image& output;
    Rect frame;
    double pixelAspect;
    double timeFrames;
    double fps;
    bool (*abort)(void*);
    void* abortContext;

    bool cancelled() const { return abort && abort(abortContext); }
};

// Calls row(y) for every y in [begin, end) from several threads and stops
// early once the host cancels. A row may only write state that belongs to it.
template <typename Row>
void forEachRow(const Context& context, int begin, int end, const Row& row) {
    parallelBands(begin, end, [&](int first, int last) {
        for (int y = first; y < last; ++y) {
            if (context.cancelled()) {
                return;
            }
            row(y);
        }
    });
}

struct LensFlareSettings {
    double centerX, centerY, brightness, size, streak, halo, ghosts, rotation;
    double red, green, blue;
    int style;
    bool flareOnly;
};

struct GlowSettings {
    double threshold, knee, radius, intensity, spread, red, green, blue;
    int blend;
    bool glowOnly;
};

struct BlurTransform {
    double shiftX, shiftY, zoom, angle, centerX, centerY;
};

struct BlurSettings {
    double shutterDuration, shutterShift, exposureBias;
    int samples;
    Edge edge;
    bool blurOnly;
    BlurTransform (*evaluate)(void*, double);
    void* evalContext;
};

struct DamageSettings {
    double amount, frequency, displacement, chroma, blocks, scanlines, noise, dropout;
    int seed;
};

struct RaysSettings {
    double centerX, centerY, length, brightness, threshold, softness, decay;
    double red, green, blue;
    int samples;
    bool raysOnly;
};

struct ZapSettings {
    double startX, startY, endX, endY, bend, brightness, coreWidth, glowRadius;
    double jaggedness, branching, speed, red, green, blue;
    int bolts, detail, seed;
    bool boltOnly;
};

struct PixelSortSettings {
    double low, high, angle, mix, minLength, maxLength, fragmentation;
    int key, seed;
    bool descending, outside;
};

struct DefocusSettings {
    double radius, roundness, rotation, aspect, highlightThreshold, highlightBoost, mix;
    int blades, samples;
    Edge edge;
};

struct LightLeakSettings {
    double centerX, centerY, brightness, size, softness, angle, speed, drift, phase;
    double red, green, blue;
    int style, blend, seed;
    bool leakOnly;
};

struct FilmSettings {
    double exposure, contrast, fade, saturation, warmth, grain, grainSize;
    double halation, halationRadius, vignette, mix;
    int profile, seed;
};

void renderLensFlare(const Context&, const LensFlareSettings&);
void renderGlow(const Context&, const GlowSettings&);
void renderBlurMoCurves(const Context&, const BlurSettings&);
void renderDigitalDamage(const Context&, const DamageSettings&);
void renderRays(const Context&, const RaysSettings&);
void renderUltraZap(const Context&, const ZapSettings&);
void renderPixelSort(const Context&, const PixelSortSettings&);
void renderRackDefocus(const Context&, const DefocusSettings&);
void renderLightLeak(const Context&, const LightLeakSettings&);
void renderFilmEffect(const Context&, const FilmSettings&);

} // namespace diamond::fx
