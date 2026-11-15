#include "NewWarp.h"

namespace diamond::fx {
namespace {

constexpr double kTurn = 6.28318530717958647693;

double bounded(double value, double minimum, double maximum, double fallback) {
    return std::isfinite(value) ? std::clamp(value, minimum, maximum) : fallback;
}

double radians(double degrees) {
    return std::isfinite(degrees) ? std::remainder(degrees, 360.0) * (kTurn / 360.0) : 0.0;
}

double aspect(const Context& context) {
    return bounded(context.pixelAspect, 1e-3, 1e3, 1.0);
}

double frameCenter(int low, int high) {
    return (static_cast<double>(low) + high - 1.0) * 0.5;
}

Pixel blend(Pixel dry, Pixel wet, float mix) {
    const float dryWeight = 1.0f - mix;
    return {dry.r * dryWeight + wet.r * mix, dry.g * dryWeight + wet.g * mix,
            dry.b * dryWeight + wet.b * mix, dry.a * dryWeight + wet.a * mix};
}

template <typename Transform>
void walk(const Context& context, float mix, bool identity, const Transform& transform) {
    const Rect window = context.output.bounds;
    if (width(context.source.bounds) == 0 || height(context.source.bounds) == 0) {
        return;
    }
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            if (identity || mix == 0.0f) {
                context.output.pixel(x, y) = context.source.read(x, y);
                continue;
            }
            const Pixel wet = transform(x, y);
            context.output.pixel(x, y) = mix == 1.0f ? wet : blend(context.source.read(x, y), wet, mix);
        }
    });
}

float lensLuma(Pixel sample) {
    // The lens is the image over black, so transparent source pixels cannot
    // produce distortion from hidden straight-alpha color.
    return 0.2126f * sample.r + 0.7152f * sample.g + 0.0722f * sample.b;
}

double gradient(float positive, float negative) {
    const double difference = static_cast<double>(positive) - negative;
    return std::isfinite(difference) ? std::clamp(difference, -1.0, 1.0) : 0.0;
}

} // namespace

void renderDistortRGB(const Context& context, const DistortRGBSettings& settings) {
    const double frameHeight = height(context.frame);
    const double pixelAspect = aspect(context);
    const double shift = frameHeight * 0.01 * bounded(settings.amount, -25, 25, 2.5);
    const double detail = frameHeight * 0.01 * bounded(settings.detail, 0.05, 5, 0.65);
    const double probeY = std::max(0.5, detail);
    const double probeX = probeY / pixelAspect;
    const double angle = radians(settings.direction);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double red = bounded(settings.red, -3, 3, 1);
    const double green = bounded(settings.green, -3, 3, 0.35);
    const double blue = bounded(settings.blue, -3, 3, -0.8);
    const auto mix = static_cast<float>(bounded(settings.mix, 0, 1, 1));
    const bool identity = shift == 0.0 || (red == 0.0 && green == 0.0 && blue == 0.0);
    const Image& source = context.source;

    walk(context, mix, identity, [&](int x, int y) {
        const double gx = gradient(lensLuma(source.sample(x + probeX, y, Edge::Clamp)),
                                   lensLuma(source.sample(x - probeX, y, Edge::Clamp)));
        const double gy = gradient(lensLuma(source.sample(x, y + probeY, Edge::Clamp)),
                                   lensLuma(source.sample(x, y - probeY, Edge::Clamp)));
        const double dx = shift * (cosine * gx - sine * gy) / pixelAspect;
        const double dy = shift * (sine * gx + cosine * gy);
        if (dx == 0.0 && dy == 0.0) {
            return source.read(x, y);
        }

        const auto warped = [&](double strength) {
            return strength == 0.0 ? source.read(x, y)
                : source.sample(x - dx * strength, y - dy * strength, settings.edge);
        };
        const Pixel r = warped(red);
        const Pixel g = green == red ? r : warped(green);
        const Pixel b = blue == red ? r : (blue == green ? g : warped(blue));
        // Each separated component already carries its own coverage. The
        // output alpha must cover their union when offsets expose edges.
        return Pixel{r.r, g.g, b.b, std::max({r.a, g.a, b.a})};
    });
}

void renderWarpFishEye(const Context& context, const WarpFishEyeSettings& settings) {
    const double pixelAspect = aspect(context);
    const double frameWidth = width(context.frame);
    const double frameHeight = height(context.frame);
    const double centerX = context.frame.x1 - 0.5 + frameWidth * bounded(settings.centerX, -1, 2, 0.5);
    const double centerY = context.frame.y1 - 0.5 + frameHeight * bounded(settings.centerY, -1, 2, 0.5);
    const double diameter = std::max(1.0, std::min(frameWidth * pixelAspect, frameHeight));
    const double xRadiusScale = 2.0 * pixelAspect / diameter;
    const double yRadiusScale = 2.0 / diameter;
    const double zoom = bounded(settings.zoom, 0.25, 4, 1);
    const double amount = bounded(settings.amount, -2, 2, 0.6);
    const auto mix = static_cast<float>(bounded(settings.mix, 0, 1, 1));

    walk(context, mix, amount == 0.0 && zoom == 1.0, [&](int x, int y) {
        const double dx = x - centerX;
        const double dy = y - centerY;
        const double nx = dx * xRadiusScale;
        const double ny = dy * yRadiusScale;
        const double radiusSquared = nx * nx + ny * ny;
        // Smooth radial inverse mapping. Within the signed amount range its
        // radial derivative stays nonnegative, so it cannot fold the image.
        const double scale = (amount == 0.0 ? 1.0 : std::exp(-amount / (1.0 + radiusSquared))) / zoom;
        return context.source.sample(centerX + dx * scale, centerY + dy * scale, settings.edge);
    });
}

void renderWarpWaves(const Context& context, const WarpWavesSettings& settings) {
    const double frameHeight = std::max(1, height(context.frame));
    const double pixelAspect = aspect(context);
    const double amplitude = frameHeight * 0.01 * bounded(settings.amplitude, 0, 25, 2);
    const double frequency = bounded(settings.frequency, 0.1, 64, 4) * kTurn / frameHeight;
    const double angle = radians(settings.direction);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double speed = bounded(settings.speed, -12, 12, 0.5);
    const double cross = bounded(settings.crossWave, 0, 1, 0.25);
    const double fps = bounded(context.fps, 1e-3, 1e6, 30);
    const double frameTime = std::isfinite(context.timeFrames) ? context.timeFrames / fps : 0.0;
    const double cycles = frameTime * speed;
    const double phase = radians(settings.phase) - (std::isfinite(cycles) ? std::remainder(cycles, 1.0) * kTurn : 0.0);
    const double centerX = frameCenter(context.frame.x1, context.frame.x2);
    const double centerY = frameCenter(context.frame.y1, context.frame.y2);
    const auto mix = static_cast<float>(bounded(settings.mix, 0, 1, 1));

    walk(context, mix, amplitude == 0.0, [&](int x, int y) {
        const double dx = (x - centerX) * pixelAspect;
        const double dy = y - centerY;
        const double longitudinal = dx * cosine + dy * sine;
        const double transverse = -dx * sine + dy * cosine;
        const double mainWave = amplitude * std::sin(longitudinal * frequency + phase);
        const double secondWave = cross == 0.0 ? 0.0
            : cross * amplitude * std::sin(transverse * frequency * 0.61 - phase + kTurn * 0.25);
        const double sampleX = x + (mainWave * sine - secondWave * cosine) / pixelAspect;
        const double sampleY = y - mainWave * cosine - secondWave * sine;
        return context.source.sample(sampleX, sampleY, settings.edge);
    });
}

} // namespace diamond::fx
