#include "Transitions05.h"

#include <array>
#include <cstdint>

namespace diamond::fx {
namespace {

constexpr double pi = 3.14159265358979323846;
constexpr int maxSamples = 32;

double bounded(double value, double minimum, double maximum, double fallback)
{
    return std::isfinite(value) ? std::clamp(value, minimum, maximum) : fallback;
}

double smooth(double value)
{
    value = std::clamp(value, 0.0, 1.0);
    return value * value * (3.0 - 2.0 * value);
}

double eased(double value, double strength)
{
    const double smoothstep = value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
    return value + (smoothstep - value) * strength;
}

float interpolate(float first, float second, float amount)
{
    return first * (1.0f - amount) + second * amount;
}

Pixel blend(Pixel first, Pixel second, float amount)
{
    if (amount <= 0.0f) {
        return first;
    }
    if (amount >= 1.0f) {
        return second;
    }
    return {interpolate(first.r, second.r, amount), interpolate(first.g, second.g, amount),
        interpolate(first.b, second.b, amount), interpolate(first.a, second.a, amount)};
}

void accumulate(Pixel& total, Pixel value, float weight)
{
    total.r += value.r * weight;
    total.g += value.g * weight;
    total.b += value.b * weight;
    total.a += value.a * weight;
}

template<class Sample>
void draw(const Context& context, Image& output, const Sample& sample)
{
    const Rect window = output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        for (int x = window.x1; x < window.x2; ++x) {
            output.pixel(x, y) = sample(x, y);
        }
    });
}

void dissolve(const Context& context, const Image& to, float mix, Image& output)
{
    if (mix <= 0.0f || mix >= 1.0f) {
        const Image& endpoint = mix <= 0.0f ? context.source : to;
        draw(context, output, [&](int x, int y) { return endpoint.read(x, y); });
        return;
    }
    draw(context, output, [&](int x, int y) { return blend(context.source.read(x, y), to.read(x, y), mix); });
}

struct Point {
    double x;
    double y;
};

struct Geometry {
    Rect frame;
    double width;
    double height;
    double aspect;
    Point center;
};

Geometry geometry(const Context& context)
{
    const Rect frame = context.frame;
    const double width = static_cast<double>(frame.x2) - frame.x1;
    const double height = static_cast<double>(frame.y2) - frame.y1;
    return {frame, width, height, bounded(context.pixelAspect, 0.01, 100.0, 1.0),
        {frame.x1 + (width - 1.0) * 0.5, frame.y1 + (height - 1.0) * 0.5}};
}

Pixel sampleWithinFrame(const Image& image, const Geometry& bounds, double x, double y)
{
    // Clamp at the logical frame so a smaller source region keeps its transparency.
    x = std::clamp(x, static_cast<double>(bounds.frame.x1), bounds.frame.x2 - 1.0);
    y = std::clamp(y, static_cast<double>(bounds.frame.y1), bounds.frame.y2 - 1.0);
    return image.sample(x, y, Edge::Transparent);
}

void dissolveGlow(const Context& context, const Image& to, float mix, double envelope,
                  const TransitionSettings& settings, const Geometry& bounds)
{
    const double intensity = envelope * bounded(settings.intensity, 0.0, 4.0, 1.0);
    const double red = bounded(settings.red, 0.0, 1.0, 1.0);
    const double green = bounded(settings.green, 0.0, 1.0, 0.85);
    const double blue = bounded(settings.blue, 0.0, 1.0, 0.65);
    if (intensity <= 1.0e-6 || (red == 0.0 && green == 0.0 && blue == 0.0)) {
        dissolve(context, to, mix, context.output);
        return;
    }

    const double radius = bounded(settings.radius, 0.0, 30.0, 3.5);
    const double vertical = radius * bounds.height / 100.0;
    const int xMargin = static_cast<int>(std::min(bounds.width, std::ceil(3.0 * vertical / bounds.aspect + 2.0)));
    const int yMargin = static_cast<int>(std::min(bounds.height, std::ceil(3.0 * vertical + 2.0)));
    const auto window = context.output.bounds;
    auto grow = [](int position, int offset, int boundary, bool lower) {
        const auto shifted = static_cast<std::int64_t>(position) + offset;
        return static_cast<int>(lower ? std::max<std::int64_t>(shifted, boundary) : std::min<std::int64_t>(shifted, boundary));
    };
    // Match the three-radius support of renderGlow's narrow and broad passes.
    const Rect sourceArea{grow(window.x1, -xMargin, bounds.frame.x1, true),
        grow(window.y1, -yMargin, bounds.frame.y1, true), grow(window.x2, xMargin, bounds.frame.x2, false),
        grow(window.y2, yMargin, bounds.frame.y2, false)};
    Image mixed(sourceArea);
    dissolve(context, to, mix, mixed);
    if (context.cancelled()) {
        return;
    }
    const GlowSettings glow{bounded(settings.threshold, 0.0, 4.0, 0.55), 0.18, radius,
        intensity, 0.40, red, green, blue, 0, false};
    const Context glowContext{mixed, context.output, bounds.frame, bounds.aspect,
        context.timeFrames, context.fps, context.abort, context.abortContext};
    renderGlow(glowContext, glow);
}

std::uint32_t hash(std::uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

double phase(std::uint32_t seed, std::uint32_t channel)
{
    const auto bits = hash(seed + 0x9e3779b9u * (channel + 1));
    return static_cast<double>(bits >> 8) * (2.0 * pi / 16777216.0);
}

struct ShakeTap {
    Point shift;
    double cosine;
    double sine;
    float mix;
    float weight;
};

ShakeTap shakeTap(double time, float mix, float weight, const Geometry& bounds, double amount, double tilt,
                  double frequency, const std::array<double, 6>& phases)
{
    const double envelope = std::sin(pi * time);
    const double cycle = time * 2.0 * pi * frequency;
    const auto wave = [&](std::size_t first) {
        return 0.68 * std::sin(cycle + phases[first]) + 0.32 * std::sin(cycle * 1.79 + phases[first + 1]);
    };
    const Point shift{bounds.width * amount * envelope * wave(0),
        bounds.width * bounds.aspect * amount * envelope * 0.62 * wave(2)};
    const double rotation = tilt * envelope * wave(4);
    const double cosine = std::cos(rotation);
    const double sine = std::sin(rotation);
    const double physicalHeight = bounds.height;
    const double physicalWidth = bounds.width * bounds.aspect;
    // Zoom the pixel footprint around the center to keep rotated corners inside the frame.
    const double overscanX = std::abs(cosine) + std::abs(sine) * physicalHeight / physicalWidth
        + 2.0 * std::abs(shift.x) / bounds.width;
    const double overscanY = std::abs(cosine) + std::abs(sine) * physicalWidth / physicalHeight
        + 2.0 * std::abs(shift.y) / bounds.height;
    const double inverseZoom = 1.0 / std::max({1.0, overscanX, overscanY});
    return {{shift.x * inverseZoom, shift.y * inverseZoom}, cosine * inverseZoom, sine * inverseZoom, mix, weight};
}

double exposureSpan(double progress, double shutter, double maximum)
{
    const double endDistance = std::min(progress, 1.0 - progress);
    return std::min(shutter * maximum, 1.8 * endDistance);
}

double tapOffset(int tap, int count)
{
    return (tap + 0.5) / count - 0.5;
}

void dissolveShake(const Context& context, const Image& to, double progress, double easing,
                   const TransitionSettings& settings, const Geometry& bounds)
{
    const double amount = bounded(settings.shakeAmount, 0.0, 15.0, 1.75) / 100.0;
    const double tilt = bounded(settings.tilt, 0.0, 15.0, 2.0) * pi / 180.0;
    const double frequency = bounded(settings.frequency, 0.0, 30.0, 5.0);
    if (frequency == 0.0 || (amount == 0.0 && tilt == 0.0)) {
        dissolve(context, to, static_cast<float>(eased(progress, easing)), context.output);
        return;
    }

    const double shutter = bounded(settings.shutter, 0.0, 1.0, 0.45);
    const double span = exposureSpan(progress, shutter, std::min(0.14, 0.65 / frequency));
    const int count = span > 1.0e-6 ? std::clamp(settings.samples, 1, maxSamples) : 1;
    const float weight = 1.0f / static_cast<float>(count);
    const auto seed = static_cast<std::uint32_t>(settings.seed);
    std::array<double, 6> phases{};
    for (std::uint32_t i = 0; i < phases.size(); ++i) {
        phases[i] = phase(seed, i);
    }
    std::array<ShakeTap, maxSamples> taps{};
    for (int i = 0; i < count; ++i) {
        const double subtime = progress + span * tapOffset(i, count);
        taps[static_cast<std::size_t>(i)] = shakeTap(subtime, static_cast<float>(eased(subtime, easing)),
            weight, bounds, amount, tilt, frequency, phases);
    }

    draw(context, context.output, [&](int x, int y) {
        const double dx = x - bounds.center.x;
        const double dy = y - bounds.center.y;
        Pixel result{};
        for (int i = 0; i < count; ++i) {
            const auto& tap = taps[static_cast<std::size_t>(i)];
            const double sourceX = bounds.center.x + tap.cosine * dx + tap.sine * dy / bounds.aspect - tap.shift.x;
            const double sourceY = bounds.center.y + tap.cosine * dy - tap.sine * dx * bounds.aspect - tap.shift.y;
            const Pixel first = sampleWithinFrame(context.source, bounds, sourceX, sourceY);
            const Pixel second = sampleWithinFrame(to, bounds, sourceX, sourceY);
            accumulate(result, blend(first, second, tap.mix), tap.weight);
        }
        return result;
    });
}

struct PanTap {
    double blendEdge;
    double inverseFeather;
    Point fromOffset;
    Point toOffset;
};

void swishPan(const Context& context, const Image& to, double progress, double easing,
              const TransitionSettings& settings, const Geometry& bounds)
{
    const double angle = bounded(settings.angle, -360.0, 360.0, 0.0) * pi / 180.0;
    const double horizontal = std::cos(angle);
    const double vertical = std::sin(angle);
    const double travel = std::abs(horizontal) * bounds.width * bounds.aspect + std::abs(vertical) * bounds.height;
    const Point movement{travel * horizontal / bounds.aspect, travel * vertical};
    const Point axis{horizontal * bounds.aspect / travel, vertical / travel};
    const double feather = std::max(0.55 * std::hypot(axis.x, axis.y), 0.0075);
    const double shutter = bounded(settings.shutter, 0.0, 1.0, 0.45);
    const double span = exposureSpan(progress, shutter, 0.23);
    const int count = span > 1.0e-6 ? std::clamp(settings.samples, 1, maxSamples) : 1;
    const float weight = 1.0f / static_cast<float>(count);
    std::array<PanTap, maxSamples> taps{};
    for (int i = 0; i < count; ++i) {
        const double subtime = progress + span * tapOffset(i, count);
        const double seam = eased(subtime, 0.65 + 0.35 * easing);
        // Narrow the blend at either endpoint so the incoming or outgoing edge does not appear early.
        const double seamWidth = std::max(1.0e-8, std::min({feather, seam, 1.0 - seam}));
        taps[static_cast<std::size_t>(i)] = {seam + seamWidth, 0.5 / seamWidth, {-movement.x * seam, -movement.y * seam},
            {movement.x * (1.0 - seam), movement.y * (1.0 - seam)}};
    }

    draw(context, context.output, [&](int x, int y) {
        const double projection = 0.5 + (x - bounds.center.x) * axis.x + (y - bounds.center.y) * axis.y;
        Pixel result{};
        for (int i = 0; i < count; ++i) {
            const auto& tap = taps[static_cast<std::size_t>(i)];
            const auto incoming = static_cast<float>(smooth((tap.blendEdge - projection) * tap.inverseFeather));
            Pixel outgoingPixel{};
            Pixel incomingPixel{};
            if (incoming < 1.0f) {
                outgoingPixel = sampleWithinFrame(context.source, bounds, x + tap.fromOffset.x, y + tap.fromOffset.y);
            }
            if (incoming > 0.0f) {
                incomingPixel = sampleWithinFrame(to, bounds, x + tap.toOffset.x, y + tap.toOffset.y);
            }
            accumulate(result, blend(outgoingPixel, incomingPixel, incoming), weight);
        }
        return result;
    });
}

} // namespace

void renderTransition(TransitionKind kind, const Context& context, const Image& to, double progress,
                      const TransitionSettings& settings)
{
    if (width(context.output.bounds) == 0 || height(context.output.bounds) == 0) {
        return;
    }
    progress = bounded(progress, 0.0, 1.0, 0.0);
    if (progress <= 0.0 || progress >= 1.0) {
        dissolve(context, to, static_cast<float>(progress), context.output);
        return;
    }

    const Geometry bounds = geometry(context);
    const double easing = bounded(settings.ease, 0.0, 1.0, 0.5);
    const auto mix = static_cast<float>(eased(progress, easing));
    if (!(bounds.width > 0.0 && bounds.height > 0.0)) {
        dissolve(context, to, mix, context.output);
        return;
    }
    switch (kind) {
    case TransitionKind::Dissolve:
        dissolve(context, to, mix, context.output);
        break;
    case TransitionKind::DissolveGlow: {
        const double envelope = std::sin(pi * progress);
        dissolveGlow(context, to, mix, envelope * envelope, settings, bounds);
        break;
    }
    case TransitionKind::DissolveShake:
        dissolveShake(context, to, progress, easing, settings, bounds);
        break;
    case TransitionKind::SwishPan:
        swishPan(context, to, progress, easing, settings, bounds);
        break;
    }
}

} // namespace diamond::fx
