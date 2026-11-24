import Foundation

/// The parts of a captured frame the analysis needs, as plain values (no pixel buffers).
struct FrameGeometry {
    var imageWidth: Int
    var imageHeight: Int
    /// Intrinsics in pixels of the captured image.
    var intrinsicsImage: Intrinsics
    var lensPosition: Double
    /// LiDAR depth in metres, row-major. Used only for the reference reading.
    var lidar: [Float]
    var lidarWidth: Int
    var lidarHeight: Int
}

/// Where the depth used for measuring came from.
enum DepthSource: String, Equatable {
    case model = "model depth"
    case focusDistance = "focus distance"
    case none = "none"
}

/// Everything shown and logged for one model run on one frame.
struct InferenceResult: Equatable {
    var modelName: String
    var modelDisplayName: String
    /// False when the model found no object of at least the minimum size.
    var found: Bool
    var classId: Int
    var classLabel: String
    /// Row-major indices of the measured component, at the model's resolution.
    var pixels: [Int]
    var maskWidth: Int
    var maskHeight: Int
    /// Whether the tap landed on the measured component (otherwise the largest was taken).
    var containsTap: Bool
    var measurement: ObjectMeasurement?
    var depthSource: DepthSource
    /// LiDAR distance over the same pixels. Reference only: not used by the model.
    var lidarDistanceM: Double?
    var focusDistanceM: Double?
    var lensPosition: Double
    /// Time spent inside the model, in milliseconds.
    var latencyMs: Double
    /// The centre-square crop that turned the frame into the model input.
    var crop: CenterSquareCrop
    /// Which model produced this, and how it was run. Filled in by the caller that knows the
    /// library entry; logged with every result so runs of different models never mix.
    var modelVersion = ""
    var architecture = ""
    var computeUnits = ""

    var areaPx: Int { pixels.count }

    /// Predicted minus LiDAR distance, in metres.
    var distanceDifferenceM: Double? {
        guard let measurement, let lidarDistanceM else { return nil }
        return measurement.distanceM - lidarDistanceM
    }

    /// Centre of the component in normalized capture-device coordinates (0...1 across the
    /// full sensor image), for placing the label on the preview.
    var centroid: (x: Double, y: Double)? {
        guard !pixels.isEmpty, maskWidth > 0, maskHeight > 0 else { return nil }
        var sumU = 0.0, sumV = 0.0
        for index in pixels {
            sumU += Double(index % maskWidth)
            sumV += Double(index / maskWidth)
        }
        let n = Double(pixels.count)
        let source = crop.toSource(x: sumU / n, y: sumV / n)
        return ((source.x + 0.5) / Double(crop.sourceWidth), (source.y + 0.5) / Double(crop.sourceHeight))
    }
}

/// Turns a raw model output into a measured object. Foundation-only, so it is testable without
/// a camera or a model file.
enum InferenceAnalysis {
    static let minimumPixels = 30

    /// - `tapImagePoint`: the tap in pixels of the captured (uncropped) image, or nil to take the
    ///   largest component.
    /// Throws for an unsupported crop or when image and depth map have different aspect ratios.
    static func analyse(output: ModelOutput, descriptor: ModelDescriptor, geometry: FrameGeometry,
                        focusDistanceM: Double?, tapImagePoint: (x: Double, y: Double)?,
                        latencyMs: Double, minPixels: Int = minimumPixels) throws -> InferenceResult {
        try CenterSquareCrop.validate(name: descriptor.crop)
        try CenterSquareCrop.checkSameAspect(imageWidth: geometry.imageWidth, imageHeight: geometry.imageHeight,
                                             depthWidth: geometry.lidarWidth, depthHeight: geometry.lidarHeight)
        let crop = CenterSquareCrop(sourceWidth: geometry.imageWidth, sourceHeight: geometry.imageHeight,
                                    outputWidth: output.width, outputHeight: output.height)
        var result = InferenceResult(
            modelName: descriptor.name, modelDisplayName: descriptor.displayName, found: false,
            classId: 0, classLabel: "", pixels: [], maskWidth: output.width, maskHeight: output.height,
            containsTap: false, measurement: nil, depthSource: .none, lidarDistanceM: nil,
            focusDistanceM: focusDistanceM, lensPosition: geometry.lensPosition, latencyMs: latencyMs,
            crop: crop)

        let components = ConnectedComponents.components(
            classMap: output.classMap, width: output.width, height: output.height, minPixels: minPixels)
        var tapIndex: Int?
        if let tapImagePoint, crop.contains(x: tapImagePoint.x, y: tapImagePoint.y) {
            let point = crop.toOutput(x: tapImagePoint.x, y: tapImagePoint.y)
            let u = min(max(Int(point.x.rounded()), 0), output.width - 1)
            let v = min(max(Int(point.y.rounded()), 0), output.height - 1)
            tapIndex = v * output.width + u
        }
        guard let chosen = ConnectedComponents.select(components, tapIndex: tapIndex) else { return result }

        result.found = true
        result.classId = Int(chosen.component.classId)
        result.classLabel = descriptor.label(forClassId: result.classId)
        result.pixels = chosen.component.pixels
        result.containsTap = chosen.containsTap
        // No depth output: the calibrated focus distance stands in as every pixel's depth.
        result.measurement = ObjectGeometry.measure(
            pixels: chosen.component.pixels, mapWidth: output.width, depth: output.depth,
            fallbackDistanceM: focusDistanceM, intrinsics: crop.intrinsics(geometry.intrinsicsImage))
        if result.measurement != nil {
            result.depthSource = output.depth != nil ? .model : .focusDistance
        }
        result.lidarDistanceM = ObjectGeometry.lidarDistance(
            pixels: chosen.component.pixels, mapWidth: output.width, mapHeight: output.height,
            lidar: geometry.lidar, lidarWidth: geometry.lidarWidth, lidarHeight: geometry.lidarHeight)
        return result
    }
}
