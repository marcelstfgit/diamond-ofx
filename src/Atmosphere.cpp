#include "Effects.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace diamond::fx {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxZapSegments = 256;

struct Point {
    double x, y;
};

struct Color {
    float r, g, b;
};

Point operator+(Point a, Point b) { return {a.x + b.x, a.y + b.y}; }
Point operator-(Point a, Point b) { return {a.x - b.x, a.y - b.y}; }
Point operator*(Point point, double factor) { return {point.x * factor, point.y * factor}; }
double dot(Point a, Point b) { return a.x * b.x + a.y * b.y; }

Point normalizedPoint(Rect frame, double x, double y, double aspect) {
    return {(frame.x1 + x * std::max(0, width(frame) - 1)) * aspect,
            frame.y1 + y * std::max(0, height(frame) - 1)};
}

std::uint32_t mixBits(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

double randomSigned(std::uint32_t key) {
    return static_cast<double>(mixBits(key)) * (2.0 / 4294967295.0) - 1.0;
}

double evolvingNoise(double phase, std::uint32_t key) {
    if (!std::isfinite(phase)) {
        return randomSigned(key);
    }
    constexpr double period = 4294967296.0;
    const double cell = std::floor(phase);
    const double wrapped = std::fmod(cell, period);
    const auto index = static_cast<std::uint32_t>(wrapped < 0.0 ? wrapped + period : wrapped);
    const double fraction = phase - cell;
    const double weight = fraction * fraction * (3.0 - 2.0 * fraction);
    const double left = randomSigned(key ^ (index * 0x9e3779b9U));
    const double right = randomSigned(key ^ ((index + 1U) * 0x9e3779b9U));
    return left + (right - left) * weight;
}

double animationTime(const Context& context, double speed) {
    return context.timeFrames * speed / (context.fps > 0.0 ? context.fps : 30.0);
}

float softBell(double distanceSquared, double inverseVariance) {
    const double exponent = 0.5 * distanceSquared * inverseVariance;
    return exponent < 8.0 ? static_cast<float>(std::exp(-exponent)) : 0.0f;
}

void addColor(Color& sum, Color color, float strength) {
    sum.r += color.r * strength;
    sum.g += color.g * strength;
    sum.b += color.b * strength;
}

Pixel withLight(Pixel base, Color light, bool screen = false) {
    const float alpha = std::clamp(std::max({light.r, light.g, light.b}), 0.0f, 1.0f);
    if (screen) {
        // Keep HDR highlights already present in the source. The isolated
        // pass always retains the original amount of generated light.
        light.r = std::min(light.r, 1.0f) * (1.0f - std::clamp(base.r, 0.0f, 1.0f));
        light.g = std::min(light.g, 1.0f) * (1.0f - std::clamp(base.g, 0.0f, 1.0f));
        light.b = std::min(light.b, 1.0f) * (1.0f - std::clamp(base.b, 0.0f, 1.0f));
    }
    return {base.r + light.r, base.g + light.g, base.b + light.b,
            base.a + (1.0f - base.a) * alpha};
}

template <typename Light>
void compositeAtmosphere(const Context& context, bool only, bool screen, const Light& light) {
    const Rect area = context.output.bounds;
    forEachRow(context, area.y1, area.y2, [&](int y) {
        for (int x = area.x1; x < area.x2; ++x) {
            const Pixel base = only ? Pixel{} : context.source.read(x, y);
            context.output.pixel(x, y) = withLight(base, light(x, y), screen && !only);
        }
    });
}

struct GlowEnergy {
    float aura = 0, bloom = 0;
};

template <typename Energy>
class Raster {
public:
    Raster(Rect area, int step, double aspect)
        : area_(step == 1 ? area : paddedGridArea(area, step)), step_(step),
          spacingX_(step * aspect), stride_(width(area_)),
          cells_(static_cast<std::size_t>(stride_) * height(area_)) {}

    Energy& cell(int x, int y) {
        return cells_[index(x, y)];
    }

    const Energy& cell(int x, int y) const {
        return cells_[index(x, y)];
    }

    template <typename Deposit>
    void drawSegment(Point start, Point end, double radius, const Deposit& deposit) {
        const int x1 = bound((std::min(start.x, end.x) - radius) / spacingX_, area_.x1, area_.x2);
        const int y1 = bound((std::min(start.y, end.y) - radius) / step_, area_.y1, area_.y2);
        const int x2 = bound((std::max(start.x, end.x) + radius) / spacingX_ + 1.0,
                             area_.x1, area_.x2);
        const int y2 = bound((std::max(start.y, end.y) + radius) / step_ + 1.0,
                             area_.y1, area_.y2);
        const Point direction = end - start;
        const double lengthSquared = dot(direction, direction);
        const double inverseLengthSquared = lengthSquared > 0.0 ? 1.0 / lengthSquared : 0.0;
        for (int y = y1; y < y2; ++y) {
            for (int x = x1; x < x2; ++x) {
                const Point fromStart{x * spacingX_ - start.x, y * step_ - start.y};
                const double fraction = std::clamp(dot(fromStart, direction) * inverseLengthSquared,
                                                   0.0, 1.0);
                const Point nearest = fromStart - direction * fraction;
                deposit(cell(x, y), dot(nearest, nearest), fraction);
            }
        }
    }

    int step() const { return step_; }

private:
    static int bound(double value, int low, int high) {
        return static_cast<int>(std::clamp(std::floor(value), static_cast<double>(low),
                                           static_cast<double>(high)));
    }

    static Rect paddedGridArea(Rect area, int step) {
        const auto lower = [step](int value) {
            return static_cast<int>(std::floor(static_cast<double>(value) / step));
        };
        return {lower(area.x1), lower(area.y1), lower(area.x2 - 1) + 2, lower(area.y2 - 1) + 2};
    }

    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y - area_.y1) * stride_ + (x - area_.x1);
    }

    Rect area_;
    int step_;
    double spacingX_;
    int stride_;
    std::vector<Energy> cells_;
};

GlowEnergy sampleGlow(const Raster<GlowEnergy>& grid, int x, int y) {
    if (grid.step() == 1) {
        return grid.cell(x, y);
    }
    const double sx = static_cast<double>(x) / grid.step();
    const double sy = static_cast<double>(y) / grid.step();
    const int left = static_cast<int>(std::floor(sx));
    const int bottom = static_cast<int>(std::floor(sy));
    const auto u = static_cast<float>(sx - left);
    const auto v = static_cast<float>(sy - bottom);
    const GlowEnergy p = grid.cell(left, bottom);
    const GlowEnergy q = grid.cell(left + 1, bottom);
    const GlowEnergy r = grid.cell(left, bottom + 1);
    const GlowEnergy s = grid.cell(left + 1, bottom + 1);
    const auto bilinear = [u, v](float a, float b, float c, float d) {
        return (a + (b - a) * u) * (1.0f - v) + (c + (d - c) * u) * v;
    };
    return {bilinear(p.aura, q.aura, r.aura, s.aura),
            bilinear(p.bloom, q.bloom, r.bloom, s.bloom)};
}

struct BoltPath {
    std::array<Point, kMaxZapSegments + 1> points;
    int segments;
};

BoltPath makeBolt(Point start, Point end, double bend, double displacement,
                  int detail, double phase, std::uint32_t key) {
    const Point delta = end - start;
    const double length = std::sqrt(dot(delta, delta));
    const Point normal = length > 0.0 ? Point{-delta.y / length, delta.x / length} : Point{0, 1};
    const int segments = 1 << detail;
    BoltPath bolt{{}, segments};
    bolt.points[0] = start;
    bolt.points[segments] = end;

    // Midpoint displacement creates angular detail with stable endpoints.
    // The same random knots evolve over time, avoiding random frame resets.
    double displacementAtLevel = displacement;
    for (int span = segments; span >= 2; span /= 2) {
        const int half = span / 2;
        for (int i = half; i < segments; i += span) {
            const auto knot = static_cast<std::uint32_t>(i) * 0x51ed270bU;
            const double wobble = evolvingNoise(phase + 0.19 * i, key ^ knot);
            bolt.points[i] = (bolt.points[i - half] + bolt.points[i + half]) * 0.5
                + normal * (displacementAtLevel * wobble);
        }
        displacementAtLevel *= 0.57;
    }
    for (int i = 1; i < segments; ++i) {
        bolt.points[i] = bolt.points[i] + normal * (bend * std::sin(kPi * i / segments));
    }
    return bolt;
}

struct ZapRadii {
    double coreSigma, glowRadius, antialiasVariance;
};

double taper(double position, bool branch) {
    if (branch) {
        return std::max(0.0, 1.0 - position);
    }
    return 0.74 + 0.26 * std::sin(kPi * position);
}

bool rasterizeBolt(const Context& context, const BoltPath& bolt, const ZapRadii& radii,
                   double widthScale, float strength, bool branch, Raster<float>& cores,
                   Raster<GlowEnergy>& glows) {
    for (int i = 0; i < bolt.segments; ++i) {
        if ((i % 8 == 0) && context.cancelled()) {
            return false;
        }
        const double from = taper(static_cast<double>(i) / bolt.segments, branch);
        const double to = taper(static_cast<double>(i + 1) / bolt.segments, branch);
        const Point a = bolt.points[i];
        const Point b = bolt.points[i + 1];
        const double sigma = radii.coreSigma * widthScale;
        const double widest = sigma * std::max(from, to);
        const double reach = 4.0 * std::sqrt(widest * widest + radii.antialiasVariance);
        cores.drawSegment(a, b, reach, [&](float& pixel, double distanceSquared, double position) {
            const double profile = from + (to - from) * position;
            const double localSigma = sigma * profile;
            const double variance = localSigma * localSigma + radii.antialiasVariance;
            const float coverage = static_cast<float>(std::min(1.0, localSigma * 2.5));
            const float energy = coverage * strength * softBell(distanceSquared, 1.0 / variance);
            pixel = std::max(pixel, energy);
        });

        if (radii.glowRadius > 0.0) {
            const double glow = radii.glowRadius * (0.45 + 0.55 * widthScale);
            const double innerSigma = std::max(0.55, glow * 0.22);
            const double outerSigma = std::max(0.85, glow * 0.47);
            const double innerVariance = 1.0 / (innerSigma * innerSigma);
            const double outerVariance = 1.0 / (outerSigma * outerSigma);
            glows.drawSegment(a, b, 4.0 * outerSigma,
                              [&](GlowEnergy& pixel, double distanceSquared, double position) {
                const auto profile = static_cast<float>(from + (to - from) * position);
                const float energy = strength * profile;
                pixel.aura = std::max(pixel.aura, energy * softBell(distanceSquared, innerVariance));
                pixel.bloom = std::max(pixel.bloom, energy * softBell(distanceSquared, outerVariance));
            });
        }
    }
    return true;
}

bool drawBranches(const Context& context, const BoltPath& main, const ZapSettings& settings,
                  const ZapRadii& radii, Point direction, Point normal, double length,
                  double phase, int detail, int boltIndex, std::uint32_t key,
                  Raster<float>& cores, Raster<GlowEnergy>& glows) {
    constexpr std::array<double, 8> sites{0.29, 0.65, 0.46, 0.80, 0.19, 0.56, 0.72, 0.37};
    const int branches = boltIndex == 0 ? static_cast<int>(sites.size()) : 4;
    for (int index = 0; index < branches; ++index) {
        const auto branchKey = mixBits(key ^ (static_cast<std::uint32_t>(index + 1) * 0xd1b54a35U));
        const double visibility = std::clamp(settings.branching * branches - index, 0.0, 1.0);
        if (visibility <= 0.0) {
            break;
        }
        const int rootIndex = std::clamp(
            static_cast<int>(main.segments * (sites[index] + 0.05 * randomSigned(branchKey))),
            1, main.segments - 1);
        const Point root = main.points[rootIndex];
        const double side = randomSigned(branchKey ^ 0x28a61c07U) < 0.0 ? -1.0 : 1.0;
        const double reach = length * (0.21 + 0.078 * randomSigned(branchKey ^ 0x95893a23U));
        const double sway = 0.06 * evolvingNoise(phase * 0.69 + 3.1, branchKey ^ 0x24a56fb9U);
        const Point tip = root + direction * (reach * 0.76) + normal * (side * reach * (0.71 + sway));
        const BoltPath branch = makeBolt(root, tip, 0.13 * side * reach,
                                         0.36 * reach * settings.jaggedness,
                                         std::max(2, detail - 2), phase * 1.13, branchKey);
        const auto intensity = static_cast<float>(visibility * (boltIndex == 0 ? 0.60 : 0.40));
        if (!rasterizeBolt(context, branch, radii, 0.54, intensity, true, cores, glows)) {
            return false;
        }

        const double forkAmount = std::clamp((settings.branching - 0.32) * 2.2, 0.0, 1.0);
        if (forkAmount > 0.0 && index % 2 == 0) {
            const Point forkRoot = branch.points[branch.segments / 2];
            const Point forkTip = forkRoot + direction * (reach * 0.44) + normal * (side * reach * 0.19);
            const BoltPath fork = makeBolt(forkRoot, forkTip, -0.07 * side * reach,
                                           0.15 * reach * settings.jaggedness,
                                           std::max(2, detail - 3), phase * 1.31,
                                           branchKey ^ 0x97ee2a11U);
            if (!rasterizeBolt(context, fork, radii, 0.31, intensity * static_cast<float>(forkAmount * 0.62),
                               true, cores, glows)) {
                return false;
            }
        }
    }
    return true;
}

struct LeakProfile {
    Color hot, mid, outer;
    double width, height, curl, split;
    float strength;
};

constexpr std::array<LeakProfile, 3> kLeakProfiles{{
    {{1.00f, 0.86f, 0.48f}, {1.00f, 0.31f, 0.12f}, {0.92f, 0.13f, 0.24f},
     0.23, 0.62, 0.17, 0.12, 1.00f},  // Warm Film
    {{0.56f, 0.89f, 1.00f}, {0.34f, 0.52f, 1.00f}, {0.80f, 0.29f, 0.78f},
     0.17, 0.57, -0.13, 0.16, 0.94f},  // Cool Prism
    {{1.00f, 0.94f, 0.61f}, {1.00f, 0.63f, 0.20f}, {1.00f, 0.30f, 0.15f},
     0.32, 0.45, 0.21, 0.18, 1.06f},  // Golden Sweep
}};

float leakLayer(double x, double y, double width, double height) {
    const double u = x / width;
    const double v = y / height;
    return softBell(u * u + v * v, 1.0);
}

Color leakAt(double u, double v, const LeakProfile& profile, int style, double softness,
             double phase, double detailPhase) {
    const double warpedU = u + profile.curl * (v * v - 0.16)
        + 0.085 * std::sin(3.4 * v + phase)
        + 0.027 * std::sin(8.3 * v - 3.1 * u + detailPhase);
    const double warpedV = v + 0.12 * std::sin(2.6 * u + detailPhase);
    const double softWidth = profile.width * (0.60 + softness * 0.82);
    const double softHeight = profile.height * (0.85 + softness * 0.35);
    Color light{};
    addColor(light, profile.hot, 0.88f * leakLayer(warpedU + 0.12, warpedV + 0.08,
                                                  softWidth * 0.49, softHeight * 0.57));
    addColor(light, profile.mid, 0.70f * leakLayer(warpedU, warpedV, softWidth, softHeight));
    addColor(light, profile.outer, 0.24f * leakLayer(warpedU - profile.split, warpedV - 0.14,
                                                    softWidth * 1.63, softHeight * 0.89));
    addColor(light, profile.mid, 0.10f * leakLayer(warpedU - 0.30, warpedV + 0.28,
                                                  softWidth * 2.2, softHeight * 1.36));

    if (style == 1) {
        addColor(light, {0.27f, 1.00f, 0.79f},
                 0.22f * leakLayer(warpedU + 0.02 - 0.32 * warpedV, warpedV + 0.19,
                                   softWidth * 0.39, softHeight * 0.83));
        addColor(light, {0.93f, 0.43f, 0.76f},
                 0.18f * leakLayer(warpedU - 0.24 + 0.10 * warpedV, warpedV + 0.06,
                                   softWidth * 0.58, softHeight * 0.75));
    } else {
        addColor(light, profile.hot,
                 0.21f * leakLayer(warpedU + 0.30 * warpedV - 0.13, warpedV - 0.22,
                                   softWidth * 0.62, softHeight * 0.53));
    }
    return light;
}

}  // namespace

void renderUltraZap(const Context& context, const ZapSettings& settings) {
    const Rect area = context.output.bounds;
    if (width(area) == 0 || height(area) == 0) {
        return;
    }
    if (settings.brightness <= 0.0 || std::max({settings.red, settings.green, settings.blue}) <= 0.0) {
        compositeAtmosphere(context, settings.boltOnly, false, [](int, int) { return Color{}; });
        return;
    }

    const double aspect = std::max(0.001, context.pixelAspect);
    const double frameHeight = std::max(1, height(context.frame));
    const Point start = normalizedPoint(context.frame, settings.startX, settings.startY, aspect);
    const Point end = normalizedPoint(context.frame, settings.endX, settings.endY, aspect);
    const Point delta = end - start;
    const double length = std::sqrt(dot(delta, delta));
    const Point direction = length > 0.0 ? delta * (1.0 / length) : Point{1, 0};
    const Point normal{-direction.y, direction.x};
    const double nominalLength = std::min(length, frameHeight * 1.5);
    const double bend = frameHeight * settings.bend * 0.01;
    const double phase = animationTime(context, settings.speed);
    const int detail = std::clamp(settings.detail, 3, 8);
    const int boltCount = std::clamp(settings.bolts, 1, 5);
    const auto seed = static_cast<std::uint32_t>(settings.seed);
    const ZapRadii radii{frameHeight * settings.coreWidth * (0.01 / 2.354820045),
                        frameHeight * settings.glowRadius * 0.01,
                        (1.0 + aspect * aspect) / 16.0};
    const int glowStep = radii.glowRadius >= 38.0 ? 4 : radii.glowRadius >= 12.0 ? 2 : 1;
    Raster<float> cores(area, 1, aspect);
    Raster<GlowEnergy> glows(radii.glowRadius > 0.0 ? area : Rect{}, glowStep, aspect);
    for (int index = 0; index < boltCount; ++index) {
        const auto key = mixBits(seed ^ (static_cast<std::uint32_t>(index + 1) * 0x9e3779b9U));
        const double spread = index == 0 ? 0.0 : nominalLength * 0.13 * randomSigned(key ^ 0x15b95357U);
        const double offset = nominalLength * settings.jaggedness * (index == 0 ? 0.19 : 0.22);
        const BoltPath bolt = makeBolt(start, end, bend + spread, offset,
                                       detail, phase + 2.4 * index, key);
        const float pulse = 1.0f + static_cast<float>(0.16 * evolvingNoise(phase * 1.78, key));
        if (!rasterizeBolt(context, bolt, radii, index == 0 ? 1.0 : 0.75,
                           pulse * (index == 0 ? 1.0f : 0.75f), false, cores, glows)
            || !drawBranches(context, bolt, settings, radii, direction, normal, nominalLength,
                             phase, detail, index, key, cores, glows)) {
            return;
        }
    }

    const Color tint{static_cast<float>(settings.red), static_cast<float>(settings.green),
                     static_cast<float>(settings.blue)};
    const float peak = std::max({tint.r, tint.g, tint.b});
    const Color whiteHot{peak * 0.76f + tint.r * 0.24f, peak * 0.76f + tint.g * 0.24f,
                         peak * 0.76f + tint.b * 0.24f};
    const auto brightness = static_cast<float>(settings.brightness);
    compositeAtmosphere(context, settings.boltOnly, false, [&](int x, int y) {
        const GlowEnergy glow = radii.glowRadius > 0.0 ? sampleGlow(glows, x, y) : GlowEnergy{};
        Color light{};
        addColor(light, whiteHot, cores.cell(x, y) * brightness * 1.32f);
        addColor(light, tint, (0.63f * glow.aura + 0.24f * glow.bloom) * brightness);
        return light;
    });
}

void renderLightLeak(const Context& context, const LightLeakSettings& settings) {
    if (settings.brightness <= 0.0 || std::max({settings.red, settings.green, settings.blue}) <= 0.0) {
        compositeAtmosphere(context, settings.leakOnly, false, [](int, int) { return Color{}; });
        return;
    }
    const int style = std::clamp(settings.style, 0, static_cast<int>(kLeakProfiles.size()) - 1);
    const LeakProfile& profile = kLeakProfiles[static_cast<std::size_t>(style)];
    const auto key = static_cast<std::uint32_t>(settings.seed);
    const double phase = settings.phase + animationTime(context, settings.speed);
    const double aspect = std::max(0.001, context.pixelAspect);
    const double frameHeight = std::max(1, height(context.frame));
    Point center = normalizedPoint(context.frame, settings.centerX, settings.centerY, aspect);
    center.x += frameHeight * 0.19 * settings.drift * evolvingNoise(phase, key ^ 0xb4425b31U);
    center.y += frameHeight * 0.17 * settings.drift * evolvingNoise(phase, key ^ 0xadff158dU);
    const double shapePhase = kPi * randomSigned(key ^ 0x91f7d29fU)
        + settings.drift * 1.9 * evolvingNoise(phase * 0.80, key ^ 0xd8312f19U);
    const double detailPhase = kPi * randomSigned(key ^ 0xf48521d3U)
        + settings.drift * 1.4 * evolvingNoise(phase * 0.62, key ^ 0x6d0b5893U);
    const double angle = settings.angle * kPi / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double inverseSize = 1.0 / (frameHeight * std::max(0.05, settings.size));
    const double gain = settings.brightness * profile.strength
        * (1.0 + 0.24 * settings.drift * evolvingNoise(phase * 1.35, key ^ 0xc9f5b2dfU));
    compositeAtmosphere(context, settings.leakOnly, settings.blend == 1, [&](int x, int y) {
        const double dx = (x * aspect - center.x) * inverseSize;
        const double dy = (y - center.y) * inverseSize;
        const Color light = leakAt(cosine * dx + sine * dy, cosine * dy - sine * dx,
                                   profile, style, settings.softness, shapePhase, detailPhase);
        return Color{light.r * static_cast<float>(gain * settings.red),
                     light.g * static_cast<float>(gain * settings.green),
                     light.b * static_cast<float>(gain * settings.blue)};
    });
}

}  // namespace diamond::fx
