import Foundation

enum CropError: Error, LocalizedError, Equatable {
    case unsupportedCrop(String)
    case aspectMismatch(imageWidth: Int, imageHeight: Int, depthWidth: Int, depthHeight: Int)

    var errorDescription: String? {
        switch self {
        case .unsupportedCrop(let name):
            return "The model asks for the crop \"\(name)\"; this app version only supports \"\(CenterSquareCrop.name)\"."
        case .aspectMismatch(let imageWidth, let imageHeight, let depthWidth, let depthHeight):
            return "The image (\(imageWidth)×\(imageHeight)) and the depth map (\(depthWidth)×\(depthHeight)) "
                + "have different aspect ratios, so they cannot be cropped to the same square."
        }
    }
}

/// The one place that knows how a full sensor frame becomes the model's square input:
/// take the centre square of the frame, then resize it to the model input.
///
/// Capture and storage keep the full 4:3 frame; this is pre-processing only. Coordinates use
/// integer values as pixel centres. Foundation-only.
///
///     S = min(W, H);  x0 = (W − S) / 2;  y0 = (H − S) / 2;  s = output / S
///     u' = (u − x0 + 0.5) · s − 0.5        (and the same for v)
struct CenterSquareCrop: Equatable {
    static let name = "center_square"

    let sourceWidth: Int
    let sourceHeight: Int
    let outputWidth: Int
    let outputHeight: Int

    /// Side of the centre square, in source pixels.
    var side: Int { min(sourceWidth, sourceHeight) }
    var x0: Int { (sourceWidth - side) / 2 }
    var y0: Int { (sourceHeight - side) / 2 }
    var scaleX: Double { Double(outputWidth) / Double(max(side, 1)) }
    var scaleY: Double { Double(outputHeight) / Double(max(side, 1)) }

    /// Fails for any crop name other than "center_square".
    static func validate(name: String) throws {
        if name != Self.name { throw CropError.unsupportedCrop(name) }
    }

    /// The image and its depth map must have the same aspect ratio (within about 1%) for their
    /// centre squares to show the same region.
    static func checkSameAspect(imageWidth: Int, imageHeight: Int, depthWidth: Int, depthHeight: Int) throws {
        guard imageWidth > 0, imageHeight > 0, depthWidth > 0, depthHeight > 0 else {
            throw CropError.aspectMismatch(imageWidth: imageWidth, imageHeight: imageHeight,
                                           depthWidth: depthWidth, depthHeight: depthHeight)
        }
        let imageAspect = Double(imageWidth) / Double(imageHeight)
        let depthAspect = Double(depthWidth) / Double(depthHeight)
        if abs(imageAspect / depthAspect - 1) > 0.01 {
            throw CropError.aspectMismatch(imageWidth: imageWidth, imageHeight: imageHeight,
                                           depthWidth: depthWidth, depthHeight: depthHeight)
        }
    }

    /// Source pixel coordinates -> model input coordinates.
    func toOutput(x: Double, y: Double) -> (x: Double, y: Double) {
        ((x - Double(x0) + 0.5) * scaleX - 0.5, (y - Double(y0) + 0.5) * scaleY - 0.5)
    }

    /// Model input coordinates -> source pixel coordinates (the inverse of `toOutput`).
    func toSource(x: Double, y: Double) -> (x: Double, y: Double) {
        ((x + 0.5) / scaleX - 0.5 + Double(x0), (y + 0.5) / scaleY - 0.5 + Double(y0))
    }

    /// Whether a source point lies inside the square the model sees.
    func contains(x: Double, y: Double) -> Bool {
        let left = Double(x0) - 0.5, top = Double(y0) - 0.5
        return x >= left && x < left + Double(side) && y >= top && y < top + Double(side)
    }

    /// Intrinsics of the cropped-and-resized image.
    func intrinsics(_ k: Intrinsics) -> Intrinsics {
        Intrinsics(fx: k.fx * scaleX, fy: k.fy * scaleY,
                   cx: (k.cx - Double(x0) + 0.5) * scaleX - 0.5,
                   cy: (k.cy - Double(y0) + 0.5) * scaleY - 0.5)
    }

    /// The square in normalized capture-device coordinates (0...1 across the sensor image),
    /// for drawing the guide and placing the mask overlay on the preview.
    var normalizedRect: (minX: Double, minY: Double, maxX: Double, maxY: Double) {
        (Double(x0) / Double(sourceWidth), Double(y0) / Double(sourceHeight),
         Double(x0 + side) / Double(sourceWidth), Double(y0 + side) / Double(sourceHeight))
    }

    /// Whether a point in normalized capture-device coordinates is inside the square of an
    /// image of the given size. Uses the same pixel mapping as the saved tap points (x · W).
    static func containsDevicePoint(x: Double, y: Double, imageWidth: Int, imageHeight: Int) -> Bool {
        CenterSquareCrop(sourceWidth: imageWidth, sourceHeight: imageHeight, outputWidth: 1, outputHeight: 1)
            .contains(x: x * Double(imageWidth), y: y * Double(imageHeight))
    }

    static let outsideHint = "Move the object inside the square"
}
