import Foundation

/// Pure maths used by capture and inference. Foundation-only, so it is unit-testable without a camera.

extension Intrinsics {
    /// Intrinsics for the same view resampled from one size to another, by plain ratio
    /// (the scaling Apple documents for `intrinsicMatrixReferenceDimensions`).
    func scaled(fromWidth: Double, fromHeight: Double, toWidth: Double, toHeight: Double) -> Intrinsics {
        let sx = toWidth / fromWidth
        let sy = toHeight / fromHeight
        return Intrinsics(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy)
    }
}

enum DepthMath {
    /// Depth in metres -> millimetres for `depth_mm.png`. NaN, infinite, non-positive and
    /// out-of-range values become 0, which means "no reading".
    static func millimetres(fromMetres metres: Float) -> UInt16 {
        guard metres.isFinite, metres > 0 else { return 0 }
        let mm = (metres * 1000).rounded()
        return mm >= 65535 ? 0 : UInt16(mm)
    }

    /// Median of the valid (finite, positive) depth values in a square window centred on (x, y).
    /// `values` is row-major, `width * height` long. Returns nil if the window has no valid value.
    static func medianDepth(values: [Float], width: Int, height: Int, x: Int, y: Int, window: Int = 5) -> Double? {
        guard width > 0, height > 0, values.count >= width * height else { return nil }
        let half = max(0, window / 2)
        let centreX = min(max(x, 0), width - 1)
        let centreY = min(max(y, 0), height - 1)
        var valid: [Float] = []
        for row in max(0, centreY - half)...min(height - 1, centreY + half) {
            for column in max(0, centreX - half)...min(width - 1, centreX + half) {
                let value = values[row * width + column]
                if value.isFinite, value > 0 { valid.append(value) }
            }
        }
        guard !valid.isEmpty else { return nil }
        valid.sort()
        let middle = valid.count / 2
        if valid.count % 2 == 1 { return Double(valid[middle]) }
        return Double(valid[middle - 1] + valid[middle]) / 2
    }

    /// Map a point in image pixels to the nearest depth-map pixel, clamped to the map.
    static func depthPixel(forImageX x: Double, y: Double, imageWidth: Int, imageHeight: Int,
                           depthWidth: Int, depthHeight: Int) -> (x: Int, y: Int) {
        let dx = Int((x * Double(depthWidth) / Double(imageWidth)).rounded(.down))
        let dy = Int((y * Double(depthHeight) / Double(imageHeight)).rounded(.down))
        return (min(max(dx, 0), depthWidth - 1), min(max(dy, 0), depthHeight - 1))
    }
}
