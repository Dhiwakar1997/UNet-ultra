import CoreGraphics
import CoreVideo
import Foundation

/// Result of a tap-to-focus: where the lens ended up and whether it stopped moving in time.
struct FocusResult: Equatable {
    /// `AVCaptureDevice.lensPosition`, 0 (nearest focus) to 1 (furthest). Not a distance.
    var lensPosition: Double
    var settled: Bool
}

/// The fixed capture set-up, recorded once so capture and inference can be shown to match.
struct CaptureConfiguration: Equatable {
    var videoWidth: Int
    var videoHeight: Int
    var depthWidth: Int
    var depthHeight: Int
    /// Left at the device default; recorded so inference can confirm it runs the same way.
    var geometricDistortionCorrectionEnabled: Bool
    var videoZoomFactor: Double

    var summary: String {
        "\(videoWidth)×\(videoHeight) video, \(depthWidth)×\(depthHeight) depth, " +
            "zoom \(String(format: "%.1f", videoZoomFactor))×, GDC \(geometricDistortionCorrectionEnabled ? "on" : "off")"
    }
}

/// One synchronized RGB + LiDAR depth frame with the camera state that goes with it.
///
/// Both buffers are in the sensor's native (landscape) orientation and are never rotated, so
/// the intrinsics stay valid for them. Used by capture (phase 1) and inference (phase 6).
struct CapturedFrame: @unchecked Sendable {
    /// 32BGRA video frame.
    let image: CVPixelBuffer
    /// DepthFloat32 map in metres; NaN or 0 where LiDAR has no reading.
    let depth: CVPixelBuffer
    /// Intrinsics in pixels of `image`.
    let intrinsicsImage: Intrinsics
    /// Intrinsics in pixels of `depth`.
    let intrinsicsDepth: Intrinsics
    /// Dimensions the calibration matrix was reported for, before scaling.
    let intrinsicsReferenceWidth: Int
    let intrinsicsReferenceHeight: Int
    /// Lens position when the frame was delivered, 0 to 1.
    let lensPosition: Double
    let isAdjustingFocus: Bool
    let iso: Double
    let exposureDurationSeconds: Double
    /// Presentation time of the video frame, in seconds on the capture clock.
    let presentationTime: Double
    let date: Date

    var imageWidth: Int { CVPixelBufferGetWidth(image) }
    var imageHeight: Int { CVPixelBufferGetHeight(image) }
    var depthWidth: Int { CVPixelBufferGetWidth(depth) }
    var depthHeight: Int { CVPixelBufferGetHeight(depth) }

    /// The depth map as a row-major array of metres.
    func depthValues() -> [Float] {
        let width = depthWidth, height = depthHeight
        var values = [Float](repeating: 0, count: width * height)
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return values }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depth)
        for row in 0..<height {
            let source = base.advanced(by: row * bytesPerRow).assumingMemoryBound(to: Float32.self)
            for column in 0..<width {
                values[row * width + column] = source[column]
            }
        }
        return values
    }

    /// LiDAR distance in metres at a point given in image pixels: the median of the valid depth
    /// readings in a small window. Returns nil if the window holds no reading.
    func lidarDistance(atImagePoint point: CGPoint, window: Int = 5) -> Double? {
        let pixel = DepthMath.depthPixel(forImageX: Double(point.x), y: Double(point.y),
                                         imageWidth: imageWidth, imageHeight: imageHeight,
                                         depthWidth: depthWidth, depthHeight: depthHeight)
        return DepthMath.medianDepth(values: depthValues(), width: depthWidth, height: depthHeight,
                                     x: pixel.x, y: pixel.y, window: window)
    }
}
