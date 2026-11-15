#include "Effects.h"

#include <array>
#include <utility>

namespace diamond::fx {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct Point {
    double x, y;
};

struct Aperture {
    double xRadius, yRadius, roundness, rotation;
    int blades;
};

struct Highlights {
    float threshold, knee, boost;

    Pixel apply(Pixel pixel) const {
        if (boost == 0 || pixel.a <= 0) {
            return pixel;
        }
        const float luminance = (0.2126f * pixel.r + 0.7152f * pixel.g + 0.0722f * pixel.b) / pixel.a;
        const float t = std::clamp((luminance - threshold) / knee, 0.0f, 1.0f);
        const float gain = 1 + boost * t * t * (3 - 2 * t);
        return {pixel.r * gain, pixel.g * gain, pixel.b * gain, pixel.a};
    }
};

void add(Pixel& destination, Pixel pixel, float weight) {
    destination.r += pixel.r * weight;
    destination.g += pixel.g * weight;
    destination.b += pixel.b * weight;
    destination.a += pixel.a * weight;
}

struct Span {
    int y;
    int left, right;
    double leftFraction, rightFraction;
    double weight;
};

std::vector<Point> apertureOutline(const Aperture& aperture) {
    std::vector<Point> points;
    const int arcSegments = aperture.roundness == 0.0 ? 0 : 8;
    const double halfSector = kPi / aperture.blades;
    for (int blade = 0; blade < aperture.blades; ++blade) {
        // Negating the shape gives sampling offsets for convolution, so an
        // isolated highlight has the requested aperture orientation.
        const double angle = aperture.rotation + kPi + 2 * halfSector * blade;
        const Point vertex{std::cos(angle), std::sin(angle)};
        for (int i = 0; i <= arcSegments; ++i) {
            const double normal = arcSegments == 0 ? angle
                : angle + halfSector * (2.0 * i / arcSegments - 1.0);
            const double core = 1.0 - aperture.roundness;
            points.push_back({
                aperture.xRadius * (core * vertex.x + aperture.roundness * std::cos(normal)),
                aperture.yRadius * (core * vertex.y + aperture.roundness * std::sin(normal))
            });
        }
    }
    return points;
}

std::pair<double, double> horizontalInterval(const Aperture& aperture,
                                              const std::vector<Point>& outline, double y) {
    if (outline.empty()) {
        const double v = y / aperture.yRadius;
        const double halfWidth = aperture.xRadius * std::sqrt(std::max(0.0, 1.0 - v * v));
        return {-halfWidth, halfWidth};
    }

    std::array<double, 2> intersections{};
    int count = 0;
    Point previous = outline.back();
    for (const Point point : outline) {
        if ((point.y > y) != (previous.y > y)) {
            const double t = (y - previous.y) / (point.y - previous.y);
            intersections[count++] = previous.x + (point.x - previous.x) * t;
            if (count == 2) {
                break;
            }
        }
        previous = point;
    }
    return std::minmax(intersections[0], intersections[1]);
}

std::vector<Span> makeSpans(const Aperture& aperture) {
    const auto outline = aperture.roundness == 1.0 ? std::vector<Point>{} : apertureOutline(aperture);
    double minY = -aperture.yRadius;
    double maxY = aperture.yRadius;
    if (!outline.empty()) {
        const auto range = std::minmax_element(outline.begin(), outline.end(),
            [](Point a, Point b) { return a.y < b.y; });
        minY = range.first->y;
        maxY = range.second->y;
    }

    // Integrate aperture coverage through each pixel row using a three-point
    // Gaussian quadrature, then use fractional horizontal pixel boundaries.
    constexpr std::array<double, 3> positions{-0.7745966692414834, 0.0, 0.7745966692414834};
    constexpr std::array<double, 3> weights{5.0 / 18, 4.0 / 9, 5.0 / 18};
    const int start = static_cast<int>(std::floor(minY + 0.5));
    const int end = static_cast<int>(std::ceil(maxY + 0.5));
    std::vector<Span> spans;
    double area = 0;
    for (int y = start; y < end; ++y) {
        const double low = std::max(y - 0.5, minY);
        const double high = std::min(y + 0.5, maxY);
        if (high <= low) {
            continue;
        }
        double left = 0.5;
        double right = 0.5;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const double crossSectionY = (low + high + (high - low) * positions[i]) * 0.5;
            const auto [l, r] = horizontalInterval(aperture, outline, crossSectionY);
            left += weights[i] * l;
            right += weights[i] * r;
        }
        if (right <= left) {
            continue;
        }
        const auto first = static_cast<int>(std::floor(left));
        const auto last = static_cast<int>(std::floor(right));
        spans.push_back({y, first, last, left - first, right - last, high - low});
        area += (right - left) * (high - low);
    }
    for (Span& span : spans) {
        span.weight /= area;
    }
    return spans;
}

struct Sum {
    double r = 0, g = 0, b = 0, a = 0;

    Sum operator+(const Sum& s) const { return {r + s.r, g + s.g, b + s.b, a + s.a}; }
    Sum operator-(const Sum& s) const { return {r - s.r, g - s.g, b - s.b, a - s.a}; }
    Sum operator*(double factor) const { return {r * factor, g * factor, b * factor, a * factor}; }

    Pixel pixel(double weight) const {
        return {static_cast<float>(r * weight), static_cast<float>(g * weight),
                static_cast<float>(b * weight), static_cast<float>(a * weight)};
    }
};

class RowCache {
public:
    RowCache(const Image& source, Highlights highlights, int capacity)
        : source_(source), highlights_(highlights), rows_(capacity) {}

    const Sum* get(int y) {
        Row& row = rows_[static_cast<std::size_t>(y) % rows_.size()];
        if (row.index != y) {
            const int columns = width(source_.bounds);
            row.index = y;
            row.prefix.resize(static_cast<std::size_t>(columns) + 1);
            row.prefix[0] = {};
            for (int x = 0; x < columns; ++x) {
                const Pixel p = highlights_.apply(source_.read(source_.bounds.x1 + x, source_.bounds.y1 + y));
                row.prefix[x + 1] = row.prefix[x] + Sum{p.r, p.g, p.b, p.a};
            }
        }
        return row.prefix.data();
    }

private:
    struct Row {
        int index = -1;
        std::vector<Sum> prefix;
    };
    const Image& source_;
    Highlights highlights_;
    std::vector<Row> rows_;
};

int edgeRow(int y, int rows, Edge edge) {
    if (y >= 0 && y < rows) {
        return y;
    }
    if (edge == Edge::Transparent) {
        return -1;
    }
    if (edge == Edge::Clamp || rows == 1) {
        return std::clamp(y, 0, rows - 1);
    }
    const int period = 2 * (rows - 1);
    int phase = y % period;
    if (phase < 0) {
        phase += period;
    }
    return std::min(phase, period - phase);
}

Sum innerIntegral(const Sum* prefix, int x, double fraction) {
    return fraction == 0.0 ? prefix[x] : prefix[x] + (prefix[x + 1] - prefix[x]) * fraction;
}

Sum outerIntegral(const Sum* prefix, int columns, int x, double fraction, Edge edge) {
    if (edge == Edge::Transparent) {
        return x < 0 ? Sum{} : prefix[columns];
    }
    if (edge == Edge::Clamp || columns == 1) {
        if (x < 0) {
            return prefix[1] * (x + fraction);
        }
        return prefix[columns] + (prefix[columns] - prefix[columns - 1]) * (x - columns + fraction);
    }

    // Reflect about the first and last pixel centers, matching Image::sample.
    const int period = 2 * (columns - 1);
    int cycles = x / period;
    int phase = x % period;
    if (phase < 0) {
        phase += period;
        --cycles;
    }
    const Sum forwardAndBack = prefix[columns] + prefix[columns - 1];
    const Sum complete = (forwardAndBack - prefix[1]) * cycles;
    if (phase < columns) {
        return complete + innerIntegral(prefix, phase, fraction);
    }
    const int mirror = period + 1 - phase;
    const Sum tail = prefix[mirror] - (prefix[mirror] - prefix[mirror - 1]) * fraction;
    return complete + forwardAndBack - tail;
}

Sum integral(const Sum* prefix, int columns, int x, double fraction, Edge edge) {
    if (x >= 0 && x < columns) {
        return innerIntegral(prefix, x, fraction);
    }
    if (x == columns && fraction == 0.0) {
        return prefix[columns];
    }
    return outerIntegral(prefix, columns, x, fraction, edge);
}

void convolve(const Context& context, const Image& source, Image& output, const Aperture& aperture,
              Highlights highlights, Edge edge) {
    const auto spans = makeSpans(aperture);
    const int columns = width(source.bounds);
    const int rows = height(source.bounds);
    const int cachedRows = std::min(rows, static_cast<int>(spans.size()));
    // Each band keeps its own row cache. A band rebuilds at most one aperture
    // height of prefix rows, so bands are kept at least that tall.
    parallelBands(output.bounds.y1, output.bounds.y2, [&](int firstRow, int lastRow) {
        RowCache cache(source, highlights, cachedRows);
        for (int y = firstRow; y < lastRow; ++y) {
            if (context.cancelled()) {
                return;
            }
            for (const Span& span : spans) {
                const int sourceY = edgeRow(y - source.bounds.y1 + span.y, rows, edge);
                if (sourceY < 0) {
                    continue;
                }
                const Sum* prefix = cache.get(sourceY);
                for (int x = output.bounds.x1; x < output.bounds.x2; ++x) {
                    const int localX = x - source.bounds.x1;
                    const Sum left = integral(prefix, columns, localX + span.left, span.leftFraction, edge);
                    const Sum right = integral(prefix, columns, localX + span.right, span.rightFraction, edge);
                    add(output.pixel(x, y), (right - left).pixel(span.weight), 1.0f);
                }
            }
        }
    }, std::max(8, cachedRows));
}

struct Split {
    int index;
    float first, next;
};

Split splitFootprint(int index, int full, int reduced) {
    const double scale = static_cast<double>(reduced) / full;
    const double left = index * scale;
    const int first = std::min(static_cast<int>(left), reduced - 1);
    const float next = first + 1 < reduced
        ? static_cast<float>(std::clamp((left + scale - first - 1) / scale, 0.0, 1.0)) : 0.0f;
    return {first, 1.0f - next, next};
}

Image reduce(const Context& context, int targetWidth, int targetHeight, Highlights highlights) {
    const Image& source = context.source;
    const int columns = width(source.bounds);
    const int rows = height(source.bounds);
    const auto normalization = static_cast<float>((static_cast<double>(targetWidth) / columns)
                                               * (static_cast<double>(targetHeight) / rows));
    std::vector<Split> xFootprints(columns);
    for (int x = 0; x < columns; ++x) {
        xFootprints[x] = splitFootprint(x, columns, targetWidth);
    }

    Image reduced({0, 0, targetWidth, targetHeight});
    for (int y = 0; y < rows; ++y) {
        if (context.cancelled()) {
            return {};
        }
        const Split vertical = splitFootprint(y, rows, targetHeight);
        const float primary = vertical.first * normalization;
        const float secondary = vertical.next * normalization;
        for (int x = 0; x < columns; ++x) {
            const Pixel pixel = highlights.apply(source.read(x + source.bounds.x1, y + source.bounds.y1));
            const Split horizontal = xFootprints[x];
            add(reduced.pixel(horizontal.index, vertical.index), pixel, horizontal.first * primary);
            if (horizontal.next > 0) {
                add(reduced.pixel(horizontal.index + 1, vertical.index), pixel, horizontal.next * primary);
            }
            if (secondary > 0) {
                add(reduced.pixel(horizontal.index, vertical.index + 1), pixel, horizontal.first * secondary);
                if (horizontal.next > 0) {
                    add(reduced.pixel(horizontal.index + 1, vertical.index + 1), pixel, horizontal.next * secondary);
                }
            }
        }
    }
    return reduced;
}

double reducedCoordinate(int position, int origin, double scale) {
    return (static_cast<double>(position) - origin + 0.5) * scale - 0.5;
}

int workingSize(int size, double radius, double budget) {
    return std::max(1, static_cast<int>(std::ceil(size / std::max(1.0, radius / budget))));
}

void composite(const Context& context, float mix, const Image* reduced, double scaleX, double scaleY) {
    const Rect window = context.output.bounds;
    forEachRow(context, window.y1, window.y2, [&](int y) {
        const double sourceY = reducedCoordinate(y, context.source.bounds.y1, scaleY);
        for (int x = window.x1; x < window.x2; ++x) {
            Pixel& output = context.output.pixel(x, y);
            if (mix == 0) {
                output = context.source.read(x, y);
                continue;
            }
            if (reduced) {
                const double sourceX = reducedCoordinate(x, context.source.bounds.x1, scaleX);
                output = reduced->sample(sourceX, sourceY);
            }
            if (mix < 1) {
                Pixel mixed = context.source.read(x, y);
                mixed.r *= 1.0f - mix;
                mixed.g *= 1.0f - mix;
                mixed.b *= 1.0f - mix;
                mixed.a *= 1.0f - mix;
                add(mixed, output, mix);
                output = mixed;
            }
        }
    });
}

double bounded(double value, double lo, double hi, double fallback) {
    return std::isfinite(value) ? std::clamp(value, lo, hi) : fallback;
}

} // namespace

void renderRackDefocus(const Context& context, const DefocusSettings& settings) {
    const int columns = width(context.source.bounds);
    const int rows = height(context.source.bounds);
    if (columns == 0 || rows == 0 || width(context.output.bounds) == 0
        || height(context.output.bounds) == 0 || context.cancelled()) {
        return;
    }

    const double radius = height(context.frame) * bounded(settings.radius, 0, 20, 0) / 100.0;
    const auto mix = static_cast<float>(bounded(settings.mix, 0, 1, 1));
    if (radius < 1e-6 || mix == 0) {
        composite(context, 0, nullptr, 1, 1);
        return;
    }

    const double aspect = std::sqrt(bounded(settings.aspect, 0.25, 4, 1));
    const double pixelAspect = bounded(context.pixelAspect, 0.01, 100, 1);
    const double angle = std::isfinite(settings.rotation) ? std::remainder(settings.rotation, 360.0) : 0.0;
    Aperture aperture{radius * aspect / pixelAspect, radius / aspect,
                      bounded(settings.roundness, 0, 1, 0.25), angle * kPi / 180,
                      std::clamp(settings.blades, 3, 12)};
    const auto threshold = static_cast<float>(bounded(settings.highlightThreshold, 0, 4, 0.72));
    const auto boost = static_cast<float>(bounded(settings.highlightBoost, 0, 8, 0) * std::min(1.0, radius));
    const Highlights highlights{threshold, std::max(0.03f, threshold * 0.2f), boost};

    // Area-prefilter large apertures so the amount of work does not grow with
    // the full-resolution blur radius. Small apertures run at native resolution.
    const double budget = 2 * std::sqrt(std::clamp(settings.samples, 12, 192));
    const int workWidth = workingSize(columns, aperture.xRadius, budget);
    const int workHeight = workingSize(rows, aperture.yRadius, budget);
    if (workWidth == columns && workHeight == rows) {
        convolve(context, context.source, context.output, aperture, highlights, settings.edge);
        if (mix < 1 && !context.cancelled()) {
            composite(context, mix, nullptr, 1, 1);
        }
        return;
    }

    const Image reduced = reduce(context, workWidth, workHeight, highlights);
    if (context.cancelled()) {
        return;
    }
    const double scaleX = static_cast<double>(workWidth) / columns;
    const double scaleY = static_cast<double>(workHeight) / rows;
    aperture.xRadius *= scaleX;
    aperture.yRadius *= scaleY;
    const Rect window = context.output.bounds;
    const auto lowX = [&](int x) { return reducedCoordinate(x, context.source.bounds.x1, scaleX); };
    const auto lowY = [&](int y) { return reducedCoordinate(y, context.source.bounds.y1, scaleY); };
    Image blurred({static_cast<int>(std::floor(lowX(window.x1))), static_cast<int>(std::floor(lowY(window.y1))),
                   static_cast<int>(std::floor(lowX(window.x2 - 1))) + 2,
                   static_cast<int>(std::floor(lowY(window.y2 - 1))) + 2});
    convolve(context, reduced, blurred, aperture, Highlights{}, settings.edge);
    if (context.cancelled()) {
        return;
    }
    composite(context, mix, &blurred, scaleX, scaleY);
}

} // namespace diamond::fx
