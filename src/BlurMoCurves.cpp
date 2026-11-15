#include "Effects.h"
#include "Gpu05.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace diamond::fx {
namespace {

constexpr int kMaxExposureSamples = 64;
constexpr double kRadiansPerDegree = 0.01745329251994329577;

struct Affine {
    double xx = 1.0;
    double xy = 0.0;
    double tx = 0.0;
    double yx = 0.0;
    double yy = 1.0;
    double ty = 0.0;
};

struct ExposureSample {
    Affine inverse;
    float weight;
};

struct Exposure {
    std::array<ExposureSample, kMaxExposureSamples> samples{};
    int count = 0;
};

double finiteOr(double value, double fallback) {
    return std::isfinite(value) ? value : fallback;
}

BlurTransform readPose(const BlurSettings& settings, double time) {
    if (!settings.evaluate) {
        return {0.0, 0.0, 1.0, 0.0, 0.5, 0.5};
    }
    BlurTransform pose = settings.evaluate(settings.evalContext, time);
    pose.shiftX = finiteOr(pose.shiftX, 0.0);
    pose.shiftY = finiteOr(pose.shiftY, 0.0);
    pose.zoom = std::clamp(finiteOr(pose.zoom, 1.0), 0.05, 20.0);
    pose.angle = std::remainder(finiteOr(pose.angle, 0.0), 360.0);
    pose.centerX = finiteOr(pose.centerX, 0.5);
    pose.centerY = finiteOr(pose.centerY, 0.5);
    return pose;
}

bool sameAffine(const Affine& a, const Affine& b) {
    return a.xx == b.xx && a.xy == b.xy && a.tx == b.tx
        && a.yx == b.yx && a.yy == b.yy && a.ty == b.ty;
}

bool samePose(const BlurTransform& a, const BlurTransform& b) {
    return a.shiftX == b.shiftX && a.shiftY == b.shiftY && a.zoom == b.zoom
        && a.angle == b.angle && a.centerX == b.centerX && a.centerY == b.centerY;
}

Affine compose(const Affine& outer, const Affine& inner) {
    return {
        outer.xx * inner.xx + outer.xy * inner.yx,
        outer.xx * inner.xy + outer.xy * inner.yy,
        outer.xx * inner.tx + outer.xy * inner.ty + outer.tx,
        outer.yx * inner.xx + outer.yy * inner.yx,
        outer.yx * inner.xy + outer.yy * inner.yy,
        outer.yx * inner.tx + outer.yy * inner.ty + outer.ty
    };
}

Affine matrixFor(const BlurTransform& pose, Rect frame, double pixelAspect, bool inverse) {
    const double width = static_cast<double>(frame.x2) - frame.x1;
    const double height = static_cast<double>(frame.y2) - frame.y1;
    const double centerX = frame.x1 + width * pose.centerX - 0.5;
    const double centerY = frame.y1 + height * pose.centerY - 0.5;
    const double movedX = centerX + width * 0.01 * pose.shiftX;
    const double movedY = centerY + height * 0.01 * pose.shiftY;
    const double angle = pose.angle * kRadiansPerDegree;
    const double scale = inverse ? 1.0 / pose.zoom : pose.zoom;
    const double sine = std::sin(inverse ? -angle : angle) * scale;
    const double cosine = std::cos(angle) * scale;
    const double xy = -sine / pixelAspect;
    const double yx = sine * pixelAspect;
    const double fromX = inverse ? movedX : centerX;
    const double fromY = inverse ? movedY : centerY;
    const double toX = inverse ? centerX : movedX;
    const double toY = inverse ? centerY : movedY;

    return {cosine, xy, toX - (cosine * fromX + xy * fromY),
            yx, cosine, toY - (yx * fromX + cosine * fromY)};
}

Exposure buildExposure(const Context& context, const BlurSettings& settings) {
    const double duration = std::max(0.0, finiteOr(settings.shutterDuration, 0.0));
    const double centerTime = context.timeFrames + finiteOr(settings.shutterShift, 0.0);
    const double bias = 2.0 * std::clamp(finiteOr(settings.exposureBias, 0.5), 0.0, 1.0) - 1.0;
    const double aspect = context.pixelAspect > 0.0 ? finiteOr(context.pixelAspect, 1.0) : 1.0;
    const BlurTransform current = readPose(settings, context.timeFrames);
    const Affine currentForward = matrixFor(current, context.frame, aspect, false);

    Exposure exposure;
    if (duration == 0.0) {
        exposure.count = 1;
        exposure.samples[0] = {
            settings.blurOnly ? Affine{} : matrixFor(current, context.frame, aspect, true),
            1.0f
        };
        return exposure;
    }

    const int count = std::clamp(settings.samples, 1, kMaxExposureSamples);
    for (int i = 0; i < count; ++i) {
        const double phase = (i + 0.5) / count;
        const BlurTransform sampled = readPose(settings, centerTime + duration * (phase - 0.5));
        Affine inverse = matrixFor(sampled, context.frame, aspect, true);
        if (settings.blurOnly) {
            // For input already in the current pose, first map an output point
            // back through the sampled pose, then into current input space.
            inverse = samePose(current, sampled) ? Affine{} : compose(currentForward, inverse);
        }

        // Midpoint samples of a linear exposure ramp have mean weight one.
        const auto weight = static_cast<float>((1.0 + bias * (2.0 * phase - 1.0)) / count);
        if (exposure.count > 0 && sameAffine(exposure.samples[exposure.count - 1].inverse, inverse)) {
            exposure.samples[exposure.count - 1].weight += weight;
        } else {
            exposure.samples[exposure.count++] = {inverse, weight};
        }
    }

    if (exposure.count == 1) {
        exposure.samples[0].weight = 1.0f;
    }
    return exposure;
}

void renderSingle(const Context& context, const Affine& inverse, Edge edge) {
    const Rect window = context.output.bounds;
    const bool identity = sameAffine(inverse, Affine{});
    forEachRow(context, window.y1, window.y2, [&](int y) {
        double sampleX = inverse.xx * window.x1 + inverse.xy * y + inverse.tx;
        double sampleY = inverse.yx * window.x1 + inverse.yy * y + inverse.ty;
        for (int x = window.x1; x < window.x2; ++x) {
            context.output.pixel(x, y) = identity ? context.source.read(x, y)
                : context.source.sample(sampleX, sampleY, edge);
            sampleX += inverse.xx;
            sampleY += inverse.yx;
        }
    });
}

struct ExposureGpuSample {
    float xx, xy, tx, yx, yy, ty, weight;
};

struct ExposureGpuSettings {
    int count, edge;
};

constexpr char kExposureGpuCode[] = R"CLC(
typedef struct { float xx,xy,tx,yx,yy,ty,weight; } ExposureSample;
typedef struct { int count,edge; } Settings;

__kernel void diamond_exposure(__global const float4* source, __global float4* output,
                               GpuImageInfo image, __global const Settings* settings,
                               __global const ExposureSample* steps) {
    const int column = (int)get_global_id(0);
    const int row = (int)get_global_id(1);
    const float x = (float)(column + image.outX);
    const float y = (float)(row + image.outY);
    float4 sum = (float4)(0.0f);
    for (int i = 0; i < settings->count; ++i) {
        const ExposureSample step = steps[i];
        const float sx = step.xx*x + step.xy*y + step.tx;
        const float sy = step.yx*x + step.yy*y + step.ty;
        sum += diamond_sample(source, image, sx, sy, settings->edge) * step.weight;
    }
    output[(size_t)row * image.outWidth + column] = sum;
}
)CLC";

bool renderExposureOnGpu(const Context& context, const Exposure& exposure, Edge edge) {
    const std::size_t pixels = static_cast<std::size_t>(width(context.output.bounds))
        * height(context.output.bounds);
    if (pixels < 256 * 256 || pixels * exposure.count < 4'000'000) {
        return false;
    }

    const auto representable = [](const Affine& t) {
        const auto coefficient = [](double value) { return std::isfinite(value) && std::abs(value) < 1e7; };
        return coefficient(t.xx) && coefficient(t.xy) && coefficient(t.tx)
            && coefficient(t.yx) && coefficient(t.yy) && coefficient(t.ty);
    };
    std::array<ExposureGpuSample, kMaxExposureSamples> steps{};
    for (int i = 0; i < exposure.count; ++i) {
        const auto& sample = exposure.samples[i];
        const Affine& t = sample.inverse;
        if (!representable(t)) {
            return false;
        }
        steps[i] = {static_cast<float>(t.xx), static_cast<float>(t.xy), static_cast<float>(t.tx),
                    static_cast<float>(t.yx), static_cast<float>(t.yy), static_cast<float>(t.ty), sample.weight};
    }
    const ExposureGpuSettings settings{exposure.count, static_cast<int>(edge)};
    return runGpu05(context, kExposureGpuCode, "diamond_exposure", &settings, sizeof(settings),
                    steps.data(), exposure.count * sizeof(steps[0]));
}

} // namespace

void renderBlurMoCurves(const Context& context, const BlurSettings& settings) {
    if (context.cancelled()) {
        return;
    }
    const Exposure exposure = buildExposure(context, settings);
    if (exposure.count == 1) {
        renderSingle(context, exposure.samples[0].inverse, settings.edge);
        return;
    }
    if (renderExposureOnGpu(context, exposure, settings.edge)) {
        return;
    }

    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        std::array<double, kMaxExposureSamples> sampleX{};
        std::array<double, kMaxExposureSamples> sampleY{};
        for (int i = 0; i < exposure.count; ++i) {
            const Affine& inverse = exposure.samples[i].inverse;
            sampleX[i] = inverse.xx * window.x1 + inverse.xy * y + inverse.tx;
            sampleY[i] = inverse.yx * window.x1 + inverse.yy * y + inverse.ty;
        }
        for (int x = window.x1; x < window.x2; ++x) {
            Pixel result{};
            for (int i = 0; i < exposure.count; ++i) {
                const ExposureSample& sample = exposure.samples[i];
                const Pixel pixel = context.source.sample(sampleX[i], sampleY[i], settings.edge);
                result.r += pixel.r * sample.weight;
                result.g += pixel.g * sample.weight;
                result.b += pixel.b * sample.weight;
                result.a += pixel.a * sample.weight;
                sampleX[i] += sample.inverse.xx;
                sampleY[i] += sample.inverse.yx;
            }
            context.output.pixel(x, y) = result;
        }
    });
}

} // namespace diamond::fx
