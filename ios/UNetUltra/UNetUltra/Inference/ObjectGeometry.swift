import Foundation

/// Width and height as seen from the camera, and distance, all in metres.
struct ObjectMeasurement: Equatable {
    var widthM: Double
    var heightM: Double
    var distanceM: Double
}

/// Size and distance from a mask, per-pixel depth and pinhole intrinsics. Geometry is computed,
/// never learned. Foundation-only; mirrors `object_size_robust_m` in `python/unetultra/geometry.py`.
enum ObjectGeometry {
    /// Lower median (the smaller middle value for an even count), like `torch.median`.
    static func lowerMedian(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.sorted()[(values.count - 1) / 2]
    }

    /// Ordinary median (mean of the two middle values for an even count).
    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
    }

    /// Linearly interpolated quantile of ascending values, like numpy and torch by default.
    static func quantile(sorted: [Double], _ q: Double) -> Double {
        guard let first = sorted.first else { return .nan }
        if sorted.count == 1 { return first }
        let position = min(max(q, 0), 1) * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(lower + 1, sorted.count - 1)
        return sorted[lower] + (position - Double(lower)) * (sorted[upper] - sorted[lower])
    }

    /// Measure one object.
    ///
    /// - `pixels`: row-major indices of the object's pixels in a map `mapWidth` wide.
    /// - `depth`: per-pixel depth in metres at the same resolution, or nil if the model has no
    ///   depth output; then `fallbackDistanceM` (the calibrated focus distance) is used for every pixel.
    /// - `intrinsics`: scaled to the same resolution.
    ///
    /// Pixels whose depth is far from the object's median are measured at the median, and the
    /// extent includes one pixel footprint, because N pixel centres span N - 1 pixel widths.
    static func measure(pixels: [Int], mapWidth: Int, depth: [Float]?, fallbackDistanceM: Double?,
                        intrinsics: Intrinsics, depthBand: Double = 0.1, minBandM: Double = 0.01,
                        trim: Double = 0.005) -> ObjectMeasurement? {
        guard !pixels.isEmpty, mapWidth > 0, intrinsics.fx > 0, intrinsics.fy > 0 else { return nil }
        var z: [Double]
        if let depth {
            guard let highest = pixels.max(), highest < depth.count else { return nil }
            z = pixels.map { Double(depth[$0]) }
        } else if let fallbackDistanceM, fallbackDistanceM > 0 {
            z = [Double](repeating: fallbackDistanceM, count: pixels.count)
        } else {
            return nil
        }
        guard let distance = lowerMedian(z), distance.isFinite, distance > 0 else { return nil }
        let band = max(minBandM, depthBand * distance)
        var xs = [Double](repeating: 0, count: pixels.count)
        var ys = [Double](repeating: 0, count: pixels.count)
        for (i, index) in pixels.enumerated() {
            if !(abs(z[i] - distance) <= band) { z[i] = distance }  // also catches NaN
            let u = Double(index % mapWidth), v = Double(index / mapWidth)
            xs[i] = (u - intrinsics.cx) * z[i] / intrinsics.fx
            ys[i] = (v - intrinsics.cy) * z[i] / intrinsics.fy
        }
        xs.sort()
        ys.sort()
        let width = quantile(sorted: xs, 1 - trim) - quantile(sorted: xs, trim) + distance / intrinsics.fx
        let height = quantile(sorted: ys, 1 - trim) - quantile(sorted: ys, trim) + distance / intrinsics.fy
        return ObjectMeasurement(widthM: width, heightM: height, distanceM: distance)
    }

    /// LiDAR reference: median of the valid readings (finite, positive) under the object's pixels.
    /// Each model pixel is mapped back through the depth map's own centre-square crop to the
    /// nearest depth pixel. Nil when there is no reading. Shown and logged only; never given
    /// to the model.
    static func lidarDistance(pixels: [Int], mapWidth: Int, mapHeight: Int,
                              lidar: [Float], lidarWidth: Int, lidarHeight: Int) -> Double? {
        guard mapWidth > 0, mapHeight > 0, lidarWidth > 0, lidarHeight > 0,
              lidar.count >= lidarWidth * lidarHeight else { return nil }
        let crop = CenterSquareCrop(sourceWidth: lidarWidth, sourceHeight: lidarHeight,
                                    outputWidth: mapWidth, outputHeight: mapHeight)
        var readings: [Double] = []
        readings.reserveCapacity(pixels.count)
        for index in pixels {
            let source = crop.toSource(x: Double(index % mapWidth), y: Double(index / mapWidth))
            let du = min(max(Int(source.x.rounded()), 0), lidarWidth - 1)
            let dv = min(max(Int(source.y.rounded()), 0), lidarHeight - 1)
            let value = lidar[dv * lidarWidth + du]
            if value.isFinite, value > 0 { readings.append(Double(value)) }
        }
        return median(readings)
    }
}
