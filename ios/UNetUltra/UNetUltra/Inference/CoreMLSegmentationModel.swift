import CoreML
import CoreVideo
import Foundation

enum ModelError: Error, LocalizedError {
    case missingImageInput
    case missingMaskOutput
    case unexpectedOutputShape(String)
    case metaCountMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .missingImageInput: return "The model has no image input named \"image\"."
        case .missingMaskOutput: return "The model has no output named \"mask_logits\"."
        case .unexpectedOutputShape(let detail): return "Unexpected model output shape: \(detail)."
        case .metaCountMismatch(let expected, let got):
            return "The model expects \(expected) meta value(s) but \(got) were computed."
        }
    }
}

/// A Core ML model that follows the app's model contract:
///
/// - input `image`: RGB image; scaling is built into the model, the app only crops and resizes
/// - input `meta` (optional): float32 `[1, N]`
/// - output `mask_logits`: `[1, C, H, W]`; C == 1 is a sigmoid mask, C > 1 is softmax with class 0 background
/// - output `depth` (optional): `[1, 1, H, W]` in metres
///
/// Everything else about the model is described by its `ModelDescriptor`.
final class CoreMLSegmentationModel: SegmentationModel, @unchecked Sendable {
    static let imageInput = "image"
    static let metaInput = "meta"
    static let maskOutput = "mask_logits"
    static let depthOutput = "depth"

    let descriptor: ModelDescriptor
    let computeUnits: MLComputeUnits
    private let model: MLModel
    private let metaCount: Int?

    /// Load a compiled model (`.mlmodelc`). The descriptor is derived from the model itself unless one is given.
    init(compiledModelURL: URL, name: String, computeUnits: MLComputeUnits = .all,
         descriptor: ModelDescriptor? = nil) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let loaded = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
        let derived = try Self.deriveDescriptor(from: loaded.modelDescription, name: name)
        let metaShape = loaded.modelDescription.inputDescriptionsByName[Self.metaInput]?.multiArrayConstraint?.shape
        self.model = loaded
        self.computeUnits = computeUnits
        self.descriptor = descriptor ?? derived
        self.metaCount = metaShape?.last?.intValue
    }

    /// Read what the app needs from the model's own description and metadata.
    static func deriveDescriptor(from description: MLModelDescription, name: String) throws -> ModelDescriptor {
        guard let constraint = description.inputDescriptionsByName[imageInput]?.imageConstraint else {
            throw ModelError.missingImageInput
        }
        guard let mask = description.outputDescriptionsByName[maskOutput] else { throw ModelError.missingMaskOutput }
        let userDefined = description.metadata[.creatorDefinedKey] as? [String: String]
        let classes = ModelDescriptor.parseClasses(json: userDefined?["classes"]) ?? ModelDescriptor.defaultClasses
        let hasMeta = description.inputDescriptionsByName[metaInput] != nil
        var channels = classes.count <= 2 ? 1 : classes.count
        if let shape = mask.multiArrayConstraint?.shape, shape.count == 4 {
            channels = shape[1].intValue
        }
        return ModelDescriptor(
            name: name, displayName: name,
            inputWidth: constraint.pixelsWide, inputHeight: constraint.pixelsHigh,
            metaFeatures: hasMeta ? ModelDescriptor.defaultMetaFeatures : [],
            maskChannels: channels,
            hasDepth: description.outputDescriptionsByName[depthOutput] != nil,
            classes: classes)
    }

    func predict(image: CVPixelBuffer, meta: [Float]) throws -> ModelOutput {
        var features: [String: MLFeatureValue] = [Self.imageInput: MLFeatureValue(pixelBuffer: image)]
        if !descriptor.metaFeatures.isEmpty {
            if let metaCount, metaCount > 0, metaCount != meta.count {
                throw ModelError.metaCountMismatch(expected: metaCount, got: meta.count)
            }
            let array = try MLMultiArray(shape: [1, NSNumber(value: meta.count)], dataType: .float32)
            for (index, value) in meta.enumerated() {
                array[index] = NSNumber(value: value)
            }
            features[Self.metaInput] = MLFeatureValue(multiArray: array)
        }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
        guard let logits = output.featureValue(for: Self.maskOutput)?.multiArrayValue else {
            throw ModelError.missingMaskOutput
        }
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 4, shape[0] == 1, shape[1] >= 1 else {
            throw ModelError.unexpectedOutputShape("mask_logits \(shape)")
        }
        let channels = shape[1], height = shape[2], width = shape[3]
        let classMap = Self.classMap(logits: MultiArrayReader.floats(from: logits), channels: channels,
                                     count: width * height, logitThreshold: descriptor.logitThreshold)
        var depth: [Float]?
        if descriptor.hasDepth, let depthArray = output.featureValue(for: Self.depthOutput)?.multiArrayValue {
            let values = MultiArrayReader.floats(from: depthArray)
            guard values.count == width * height else {
                throw ModelError.unexpectedOutputShape("depth \(depthArray.shape) against mask \(shape)")
            }
            depth = values
        }
        return ModelOutput(classMap: classMap, depth: depth, width: width, height: height)
    }

    /// Logits in `[C, H*W]` order -> class id per pixel.
    static func classMap(logits: [Float], channels: Int, count: Int, logitThreshold: Float) -> [UInt8] {
        var classMap = [UInt8](repeating: 0, count: count)
        guard logits.count >= channels * count else { return classMap }
        if channels == 1 {
            for i in 0..<count where logits[i] > logitThreshold { classMap[i] = 1 }
        } else {
            for i in 0..<count {
                var best = 0
                var bestValue = logits[i]
                for c in 1..<channels where logits[c * count + i] > bestValue {
                    best = c
                    bestValue = logits[c * count + i]
                }
                classMap[i] = UInt8(min(best, 255))
            }
        }
        return classMap
    }
}

/// Copies an `MLMultiArray` into a flat row-major `[Float]`, honouring its strides and data type.
enum MultiArrayReader {
    static func floats(from array: MLMultiArray) -> [Float] {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        guard count > 0, !shape.isEmpty else { return values }
        let last = shape.count - 1
        let inner = shape[last], innerStride = strides[last]
        let outerCount = count / inner
        let dataType = array.dataType
        array.withUnsafeBytes { raw in
            var index = [Int](repeating: 0, count: last)
            for outer in 0..<outerCount {
                var base = 0
                for d in 0..<last { base += index[d] * strides[d] }
                let start = outer * inner
                switch dataType {
                case .float32:
                    let pointer = raw.bindMemory(to: Float32.self)
                    for i in 0..<inner { values[start + i] = pointer[base + i * innerStride] }
                case .double:
                    let pointer = raw.bindMemory(to: Double.self)
                    for i in 0..<inner { values[start + i] = Float(pointer[base + i * innerStride]) }
                case .float16:
                    let pointer = raw.bindMemory(to: UInt16.self)
                    for i in 0..<inner { values[start + i] = halfToFloat(pointer[base + i * innerStride]) }
                case .int32:
                    let pointer = raw.bindMemory(to: Int32.self)
                    for i in 0..<inner { values[start + i] = Float(pointer[base + i * innerStride]) }
                @unknown default:
                    break
                }
                var d = last - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
        }
        return values
    }

    /// IEEE 754 half precision bits -> Float, without needing the `Float16` type.
    static func halfToFloat(_ bits: UInt16) -> Float {
        let sign: Float = (bits & 0x8000) != 0 ? -1 : 1
        let exponent = Int((bits >> 10) & 0x1F)
        let fraction = Float(bits & 0x3FF) / 1024
        if exponent == 0 { return sign * fraction * Float(sign: .plus, exponent: -14, significand: 1) }
        if exponent == 31 { return fraction == 0 ? sign * .infinity : .nan }
        return sign * (1 + fraction) * Float(sign: .plus, exponent: exponent - 15, significand: 1)
    }
}
