#include "Effects.h"
#include "Gpu05.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace diamond::fx {
namespace {

constexpr double kRadiansPerDegree = 0.01745329251994329577;
constexpr int kMaxRaySamples = 96;

struct Color {
    float r, g, b;
};

struct Point {
    double x, y;
};

struct Ghost {
    double position;
    double radius;
    float intensity;
    Color color;
    bool aperture;
};

constexpr std::array<Ghost, 8> kGhosts{{
    {-1.12, 0.089, 0.110f, {0.24f, 0.61f, 1.00f}, false},
    {-0.78, 0.041, 0.130f, {0.61f, 0.99f, 0.53f}, true},
    {-0.55, 0.017, 0.110f, {1.00f, 0.36f, 0.26f}, false},
    {-0.28, 0.064, 0.074f, {0.68f, 0.41f, 1.00f}, false},
    {-0.12, 0.023, 0.115f, {0.44f, 0.77f, 1.00f}, true},
    { 0.14, 0.012, 0.125f, {1.00f, 0.79f, 0.33f}, false},
    { 0.45, 0.042, 0.082f, {1.00f, 0.45f, 0.22f}, true},
    { 1.36, 0.081, 0.090f, {0.53f, 0.83f, 1.00f}, false},
}};

struct FlareProfile {
    Color core;
    Color bloom;
    Color beam;
    double coreWidth;
    double bloomWidth;
    double haloRadius;
    float ghostGain;
};

constexpr std::array<FlareProfile, 3> kProfiles{{
    {{0.86f, 0.94f, 1.00f}, {0.31f, 0.59f, 1.00f}, {0.27f, 0.56f, 1.00f},
     0.015, 0.108, 0.170, 1.00f},  // Cinematic anamorphic
    {{1.00f, 0.95f, 0.84f}, {1.00f, 0.66f, 0.32f}, {1.00f, 0.79f, 0.48f},
     0.023, 0.127, 0.135, 1.16f},  // Spherical
    {{1.00f, 0.98f, 0.94f}, {1.00f, 0.80f, 0.65f}, {1.00f, 0.89f, 0.78f},
     0.043, 0.155, 0.211, 0.31f},  // Soft bloom
}};

struct PositionedGhost {
    Point center;
    double inverseRadius;
};

Point framePoint(Rect frame, double x, double y) {
    const double width = static_cast<double>(frame.x2) - frame.x1;
    const double height = static_cast<double>(frame.y2) - frame.y1;
    return {frame.x1 + x * (width - 1.0), frame.y1 + y * (height - 1.0)};
}

float smoothStep(float low, float high, float value) {
    const float t = std::clamp((value - low) / (high - low), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float gaussian(double squaredDistance, double inverseVariance) {
    const double exponent = 0.5 * squaredDistance * inverseVariance;
    return exponent < 12.0 ? static_cast<float>(std::exp(-exponent)) : 0.0f;
}

float bell(double squaredDistance, double width) {
    return gaussian(squaredDistance, 1.0 / (width * width));
}

float beam(double along, double across, double length, double width) {
    const double alongScaled = along / length;
    const double acrossScaled = across / width;
    return gaussian(alongScaled * alongScaled + acrossScaled * acrossScaled, 1.0);
}

void addLight(Color& sum, Color color, float intensity) {
    sum.r += color.r * intensity;
    sum.g += color.g * intensity;
    sum.b += color.b * intensity;
}

Pixel compositeLight(Pixel base, Color light) {
    // Light is additive over the source. Its peak channel doubles as coverage
    // on transparent pixels, allowing the isolated pass to composite cleanly.
    const float coverage = std::clamp(std::max({light.r, light.g, light.b}), 0.0f, 1.0f);
    return {base.r + light.r, base.g + light.g, base.b + light.b,
            base.a + (1.0f - base.a) * coverage};
}

Color flareAt(Point p, Point hotspot, double xScale, double yScale, double cosine,
              double sine, const FlareProfile& profile, int style,
              const std::array<PositionedGhost, kGhosts.size()>& ghosts,
              const LensFlareSettings& settings) {
    const double x = (p.x - hotspot.x) * xScale;
    const double y = (p.y - hotspot.y) * yScale;
    const double distanceSquared = x * x + y * y;
    const double along = cosine * x + sine * y;
    const double across = cosine * y - sine * x;
    Color light{};

    addLight(light, profile.core, 1.55f * bell(distanceSquared, profile.coreWidth)
             + 0.50f * bell(distanceSquared, profile.coreWidth * 2.6));
    addLight(light, profile.bloom, 0.24f * bell(distanceSquared, profile.bloomWidth)
             + 0.040f * bell(distanceSquared, profile.bloomWidth * 2.65));

    if (settings.streak > 0.0) {
        float intensity = 0.0f;
        switch (style) {
            case 0:
                intensity = 0.48f * beam(along, across, 0.69, 0.0028)
                    + 0.21f * beam(along, across, 0.34, 0.009)
                    + 0.062f * beam(along, across, 0.48, 0.024);
                addLight(light, {0.57f, 0.93f, 1.00f}, static_cast<float>(settings.streak)
                         * 0.045f * beam(along, across - 0.009, 0.56, 0.0022));
                break;
            case 1: {
                constexpr double diagonal = 0.7071067811865475;
                intensity = 0.19f * beam(along, across, 0.25, 0.0044)
                    + 0.13f * beam(across, along, 0.21, 0.0044)
                    + 0.11f * beam(diagonal * (along + across), diagonal * (across - along),
                                   0.19, 0.0042)
                    + 0.08f * beam(diagonal * (along - across), diagonal * (along + across),
                                   0.15, 0.0042);
                break;
            }
            default:
                intensity = 0.11f * beam(along, across, 0.30, 0.025);
                break;
        }
        addLight(light, profile.beam, intensity * static_cast<float>(settings.streak));
    }

    if (settings.halo > 0.0 && distanceSquared < 0.15) {
        const double ring = std::sqrt(distanceSquared) - profile.haloRadius;
        const float intensity = static_cast<float>(settings.halo);
        addLight(light, profile.bloom, intensity * 0.075f * bell(ring * ring, 0.010));
        const double outer = ring - 0.019;
        const double inner = ring + 0.011;
        addLight(light, {1.00f, 0.46f, 0.27f}, intensity * 0.033f * bell(outer * outer, 0.007));
        addLight(light, {0.30f, 0.92f, 0.73f}, intensity * 0.029f * bell(inner * inner, 0.006));
    }

    if (settings.ghosts > 0.0) {
        const float gain = profile.ghostGain * static_cast<float>(settings.ghosts);
        for (std::size_t index = 0; index < kGhosts.size(); ++index) {
            const PositionedGhost& placed = ghosts[index];
            const double gx = (p.x - placed.center.x) * placed.inverseRadius * xScale;
            const double gy = (p.y - placed.center.y) * placed.inverseRadius * yScale;
            const double radiusSquared = gx * gx + gy * gy;
            if (radiusSquared >= 2.25) {
                continue;
            }
            const Ghost& ghost = kGhosts[index];
            double radius = std::sqrt(radiusSquared);
            if (ghost.aperture) {
                const double u = std::abs(cosine * gx + sine * gy);
                const double v = std::abs(cosine * gy - sine * gx);
                radius = std::max(v, 0.8660254037844386 * u + 0.5 * v);
            }
            const float disk = 1.0f - smoothStep(0.70f, 1.04f, static_cast<float>(radius));
            const double rim = radius - 0.93;
            const float shaped = 0.27f * disk + 0.59f * bell(rim * rim, 0.063)
                + 0.14f * bell(radiusSquared, 0.39);
            const float shading = static_cast<float>(std::clamp(0.77 + 0.18 * (gx - gy), 0.4, 1.0));
            addLight(light, ghost.color, gain * ghost.intensity * shaped * shading);
        }
    }
    return light;
}

class HighlightMask {
public:
    explicit HighlightMask(Rect bounds)
        : bounds_(bounds),
          width_(static_cast<std::size_t>(std::max(0, bounds.x2 - bounds.x1))),
          pixels_(width_ * static_cast<std::size_t>(std::max(0, bounds.y2 - bounds.y1))) {}

    const float* data() const { return pixels_.data(); }
    std::size_t byteCount() const { return pixels_.size() * sizeof(float); }

    bool fill(const Context& context, double threshold, double softness) {
        const float cutoff = static_cast<float>(threshold);
        const float knee = static_cast<float>(softness);
        const float normalization = 1.0f / std::max(0.15f, 1.0f - cutoff);
        forEachRow(context, bounds_.y1, bounds_.y2, [&](int y) {
            for (int x = bounds_.x1; x < bounds_.x2; ++x) {
                const Pixel pixel = context.source.read(x, y);
                const float luma = 0.2126f * pixel.r + 0.7152f * pixel.g + 0.0722f * pixel.b;
                const float excess = std::max(0.0f, luma - cutoff);
                const float rolloff = knee > 0.0f ? smoothStep(0.0f, knee, excess) : 1.0f;
                pixels_[index(x, y)] = excess * rolloff * normalization;
            }
        });
        return !context.cancelled();
    }

    float sample(double x, double y) const {
        if (!(x > static_cast<double>(bounds_.x1) - 1.0 && x < bounds_.x2
              && y > static_cast<double>(bounds_.y1) - 1.0 && y < bounds_.y2)) {
            return 0.0f;
        }
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const float fx = static_cast<float>(x - x0);
        const float fy = static_cast<float>(y - y0);
        if (x0 >= bounds_.x1 && x0 + 1 < bounds_.x2
            && y0 >= bounds_.y1 && y0 + 1 < bounds_.y2) {
            const std::size_t offset = index(x0, y0);
            const float top = pixels_[offset] * (1.0f - fx) + pixels_[offset + 1] * fx;
            const float bottom = pixels_[offset + width_] * (1.0f - fx)
                + pixels_[offset + width_ + 1] * fx;
            return top * (1.0f - fy) + bottom * fy;
        }
        return (read(x0, y0) * (1.0f - fx) + read(x0 + 1, y0) * fx) * (1.0f - fy)
            + (read(x0, y0 + 1) * (1.0f - fx) + read(x0 + 1, y0 + 1) * fx) * fy;
    }

private:
    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y - bounds_.y1) * width_
            + static_cast<std::size_t>(x - bounds_.x1);
    }

    float read(int x, int y) const {
        if (x < bounds_.x1 || x >= bounds_.x2 || y < bounds_.y1 || y >= bounds_.y2) {
            return 0.0f;
        }
        return pixels_[index(x, y)];
    }

    Rect bounds_;
    std::size_t width_;
    std::vector<float> pixels_;
};

template <typename Effect>
void renderLight(const Context& context, bool lightOnly, Color tint, const Effect& effect) {
    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            const Pixel base = lightOnly ? Pixel{} : context.source.read(x, y);
            const Color light = effect(x, y);
            context.output.pixel(x, y) = compositeLight(
                base, {light.r * tint.r, light.g * tint.g, light.b * tint.b});
        }
    });
}

Color gainAndTint(double brightness, double red, double green, double blue) {
    return {static_cast<float>(brightness * red), static_cast<float>(brightness * green),
            static_cast<float>(brightness * blue)};
}

struct RayGpuStep {
    float offset, weight;
};

struct RayGpuSettings {
    int maskX, maskY, maskWidth, maskHeight, count, lightOnly;
    float lightX, lightY, tintR, tintG, tintB;
    std::array<RayGpuStep, kMaxRaySamples> steps;
};

constexpr char kRaysGpuCode[] = R"CLC(
typedef struct { float offset,weight; } RayStep;
typedef struct {
    int maskX,maskY,maskWidth,maskHeight,count,lightOnly;
    float lightX,lightY,tintR,tintG,tintB;
    RayStep steps[96];
} Settings;

float read_highlight(__global const float* highlights, __global const Settings* s, int x, int y) {
    if (x < s->maskX || y < s->maskY || x >= s->maskX + s->maskWidth
        || y >= s->maskY + s->maskHeight) {
        return 0.0f;
    }
    return highlights[(size_t)(y - s->maskY)*s->maskWidth + x - s->maskX];
}

float sample_highlight(__global const float* highlights, __global const Settings* s, float x, float y) {
    if (!(x > s->maskX-1.0f && x < s->maskX+s->maskWidth
       && y > s->maskY-1.0f && y < s->maskY+s->maskHeight)) {
        return 0.0f;
    }
    const int x0 = convert_int_rtn(x);
    const int y0 = convert_int_rtn(y);
    const float u = x-x0;
    const float v = y-y0;
    if (x0 >= s->maskX && y0 >= s->maskY && x0+1 < s->maskX+s->maskWidth
        && y0+1 < s->maskY+s->maskHeight) {
        const size_t at = (size_t)(y0-s->maskY)*s->maskWidth + x0-s->maskX;
        return (highlights[at]*(1.0f-u) + highlights[at+1]*u)*(1.0f-v)
            + (highlights[at+s->maskWidth]*(1.0f-u) + highlights[at+s->maskWidth+1]*u)*v;
    }
    return (read_highlight(highlights,s,x0,y0)*(1.0f-u) + read_highlight(highlights,s,x0+1,y0)*u)*(1.0f-v)
        + (read_highlight(highlights,s,x0,y0+1)*(1.0f-u) + read_highlight(highlights,s,x0+1,y0+1)*u)*v;
}

__kernel void diamond_rays(__global const float4* source, __global float4* output,
                           GpuImageInfo image, __global const Settings* s,
                           __global const float* highlights) {
    const int column = (int)get_global_id(0);
    const int row = (int)get_global_id(1);
    const int x = column + image.outX;
    const int y = row + image.outY;
    const float towardX = s->lightX-x;
    const float towardY = s->lightY-y;
    float energy = 0.0f;
    for (int i = 0; i < s->count; ++i) {
        const RayStep step = s->steps[i];
        energy += step.weight*sample_highlight(highlights, s, x+towardX*step.offset, y+towardY*step.offset);
    }
    const float4 base = s->lightOnly ? (float4)(0.0f) : diamond_read(source, image, x, y);
    const float3 light = energy*(float3)(s->tintR,s->tintG,s->tintB);
    const float coverage = clamp(fmax(fmax(light.x,light.y),light.z),0.0f,1.0f);
    output[(size_t)row * image.outWidth + column] = (float4)(base.xyz+light,base.w+(1.0f-base.w)*coverage);
}
)CLC";

bool renderRaysOnGpu(const Context& context, const RaysSettings& settings, Rect available, Point light,
                     Color tint, const HighlightMask& highlights, int count,
                     const std::array<double, kMaxRaySamples>& offsets,
                     const std::array<float, kMaxRaySamples>& weights) {
    const std::size_t pixels = static_cast<std::size_t>(width(context.output.bounds))
        * height(context.output.bounds);
    if (pixels < 256 * 256 || pixels * count < 4'000'000 || !highlights.byteCount()) {
        return false;
    }
    RayGpuSettings gpu{available.x1, available.y1, width(available), height(available), count,
        settings.raysOnly ? 1 : 0, static_cast<float>(light.x), static_cast<float>(light.y),
        tint.r, tint.g, tint.b, {}};
    for (int i = 0; i < count; ++i) {
        gpu.steps[i] = {static_cast<float>(offsets[i]), weights[i]};
    }
    return runGpu05(context, kRaysGpuCode, "diamond_rays", &gpu, sizeof(gpu),
                    highlights.data(), highlights.byteCount());
}

}  // namespace

void renderLensFlare(const Context& context, const LensFlareSettings& settings) {
    const Point hotspot = framePoint(context.frame, settings.centerX, settings.centerY);
    const Point opticalCenter = framePoint(context.frame, 0.5, 0.5);
    const double height = static_cast<double>(context.frame.y2) - context.frame.y1;
    const double yScale = 1.0 / (std::max(1.0, height) * std::max(0.05, settings.size));
    const double xScale = yScale * context.pixelAspect;
    const double angle = settings.rotation * kRadiansPerDegree;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const int style = std::clamp(settings.style, 0, static_cast<int>(kProfiles.size()) - 1);
    const FlareProfile& profile = kProfiles[static_cast<std::size_t>(style)];
    std::array<PositionedGhost, kGhosts.size()> ghosts{};
    for (std::size_t index = 0; index < kGhosts.size(); ++index) {
        const Ghost& ghost = kGhosts[index];
        ghosts[index] = {{opticalCenter.x + (hotspot.x - opticalCenter.x) * ghost.position,
                          opticalCenter.y + (hotspot.y - opticalCenter.y) * ghost.position},
                         1.0 / ghost.radius};
    }

    const Color tint = gainAndTint(settings.brightness, settings.red, settings.green, settings.blue);
    renderLight(context, settings.flareOnly, tint, [&](int x, int y) {
        return flareAt({static_cast<double>(x), static_cast<double>(y)}, hotspot, xScale, yScale,
                       cosine, sine, profile, style, ghosts, settings);
    });
}

void renderRays(const Context& context, const RaysSettings& settings) {
    if (settings.length <= 0.0 || settings.brightness <= 0.0) {
        renderLight(context, settings.raysOnly, {}, [](int, int) { return Color{}; });
        return;
    }

    const Rect source = context.source.bounds;
    const Rect frame = context.frame;
    const Rect available{std::max(source.x1, frame.x1), std::max(source.y1, frame.y1),
                         std::min(source.x2, frame.x2), std::min(source.y2, frame.y2)};
    HighlightMask highlights(available);
    if (!highlights.fill(context, settings.threshold, settings.softness)) {
        return;
    }

    const int samples = std::clamp(settings.samples, 8, kMaxRaySamples);
    const double extent = 1.0 - std::exp(-settings.length);
    std::array<double, kMaxRaySamples> offsets{};
    std::array<float, kMaxRaySamples> weights{};
    float totalWeight = 0.0f;
    for (int index = 0; index < samples; ++index) {
        const double position = (index + 0.5) / samples;
        offsets[index] = position * extent;
        weights[index] = static_cast<float>(std::exp(-position * settings.decay));
        totalWeight += weights[index];
    }
    for (int index = 0; index < samples; ++index) {
        weights[index] /= totalWeight;
    }

    const Point light = framePoint(frame, settings.centerX, settings.centerY);
    const double intensity = settings.brightness * std::min(settings.length * 4.0, 1.0);
    const Color tint = gainAndTint(intensity, settings.red, settings.green, settings.blue);
    if (renderRaysOnGpu(context, settings, available, light, tint, highlights, samples, offsets, weights)) {
        return;
    }
    renderLight(context, settings.raysOnly, tint, [&](int x, int y) {
        const double towardX = light.x - x;
        const double towardY = light.y - y;
        float energy = 0.0f;
        for (int index = 0; index < samples; ++index) {
            const double offset = offsets[index];
            energy += weights[index]
                * highlights.sample(x + towardX * offset, y + towardY * offset);
        }
        return Color{energy, energy, energy};
    });
}

}  // namespace diamond::fx
