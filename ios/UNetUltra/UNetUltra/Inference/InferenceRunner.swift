import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

enum InferenceError: Error, LocalizedError {
    case resizeFailed

    var errorDescription: String? {
        switch self {
        case .resizeFailed: return "The camera frame could not be cropped and resized for the model."
        }
    }
}

/// Crops the centre square of a BGRA frame and resizes it to the model input. Nothing is
/// rotated: the result is in sensor orientation, like the training images.
enum FramePreprocessor {
    /// Antialiased (high-quality) downscale of the crop's square to `crop.outputWidth × outputHeight`.
    static func cropAndResize(_ source: CVPixelBuffer, crop: CenterSquareCrop) -> CVPixelBuffer? {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(source) == crop.sourceWidth, CVPixelBufferGetHeight(source) == crop.sourceHeight,
              crop.side > 0, crop.outputWidth > 0, crop.outputHeight > 0
        else { return nil }
        var created: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()]
        guard CVPixelBufferCreate(kCFAllocatorDefault, crop.outputWidth, crop.outputHeight, kCVPixelFormatType_32BGRA,
                                  attributes as CFDictionary, &created) == kCVReturnSuccess,
              let destination = created
        else { return nil }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination)
        else { return nil }
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(source)
        // Point the source buffer at the top-left pixel of the centre square.
        var sourceBuffer = vImage_Buffer(
            data: sourceBase.advanced(by: crop.y0 * sourceRowBytes + crop.x0 * 4),
            height: vImagePixelCount(crop.side), width: vImagePixelCount(crop.side), rowBytes: sourceRowBytes)
        var destinationBuffer = vImage_Buffer(
            data: destinationBase, height: vImagePixelCount(crop.outputHeight),
            width: vImagePixelCount(crop.outputWidth), rowBytes: CVPixelBufferGetBytesPerRow(destination))
        let error = vImageScale_ARGB8888(&sourceBuffer, &destinationBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        return error == kvImageNoError ? destination : nil
    }
}

/// Runs a model on a frame. The single entry point for live inference now, and for
/// freeze-frame comparison and benchmarks later.
///
/// The model receives the cropped image and the meta vector only. LiDAR depth from the frame
/// is read afterwards, purely as the reference reading.
enum InferenceRunner {
    /// - `tapImagePoint`: the tap in pixels of the frame's (uncropped) image, or nil.
    static func run(model: SegmentationModel, calibration: LensCalibration?, frame: CapturedFrame,
                    tapImagePoint: CGPoint?) throws -> InferenceResult {
        let descriptor = model.descriptor
        try CenterSquareCrop.validate(name: descriptor.crop)
        let crop = CenterSquareCrop(sourceWidth: frame.imageWidth, sourceHeight: frame.imageHeight,
                                    outputWidth: descriptor.inputWidth, outputHeight: descriptor.inputHeight)
        let focusDistance = calibration?.focusDistance(lensPosition: frame.lensPosition)
        let meta = try MetaFeatureProvider.values(
            for: descriptor.metaFeatures,
            inputs: MetaFeatureInputs(focusDistanceM: focusDistance, fxModel: crop.intrinsics(frame.intrinsicsImage).fx,
                                      lensPosition: frame.lensPosition))
        // Same colour conversion as the saved training JPEGs (camera buffer -> sRGB), then crop.
        let colourMatched = ImageEncoding.srgbPixelBuffer(from: frame.image) ?? frame.image
        guard let input = FramePreprocessor.cropAndResize(colourMatched, crop: crop) else {
            throw InferenceError.resizeFailed
        }

        let started = DispatchTime.now().uptimeNanoseconds
        let output = try model.predict(image: input, meta: meta)
        let latencyMs = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        let geometry = FrameGeometry(
            imageWidth: frame.imageWidth, imageHeight: frame.imageHeight, intrinsicsImage: frame.intrinsicsImage,
            lensPosition: frame.lensPosition, lidar: frame.depthValues(),
            lidarWidth: frame.depthWidth, lidarHeight: frame.depthHeight)
        return try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor, geometry: geometry, focusDistanceM: focusDistance,
            tapImagePoint: tapImagePoint.map { (x: Double($0.x), y: Double($0.y)) }, latencyMs: latencyMs)
    }
}
