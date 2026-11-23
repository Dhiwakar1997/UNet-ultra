import CoreVideo
import Foundation

/// What the app needs to know about a model to run it: the "model card". Foundation-only.
///
/// Pre-processing and the meta vector are driven entirely by this description, so no
/// model-specific code lives in the app. Today it is derived from the loaded Core ML model;
/// later it can be decoded from a card file shipped with the model.
struct ModelDescriptor: Codable, Equatable {
    var name: String
    var displayName: String
    var inputWidth: Int
    var inputHeight: Int
    /// Names of the meta features, in the order the model expects. Empty for an image-only model.
    var metaFeatures: [String]
    /// 1 = sigmoid mask; more = softmax over classes with channel 0 as background.
    var maskChannels: Int
    /// False when the model has no depth output; the calibrated focus distance is used instead.
    var hasDepth: Bool
    /// Class names; index is the class id and index 0 is "background".
    var classes: [String]
    var maskThreshold: Double = 0.5
    /// How a frame becomes the model input. Only "center_square" is supported: the centre square
    /// of the frame, resized to the input size.
    var crop: String = CenterSquareCrop.name

    static let defaultMetaFeatures = [MetaFeatureName.logFocusDistance, MetaFeatureName.fxNorm]
    static let defaultClasses = ["background", "object"]

    /// Class list from the model's metadata string, a JSON array such as `["background", "airpods_case"]`.
    static func parseClasses(json: String?) -> [String]? {
        guard let json, let data = json.data(using: .utf8),
              let classes = try? JSONDecoder().decode([String].self, from: data), !classes.isEmpty
        else { return nil }
        return classes
    }

    /// Label for a class id, tolerant of ids outside the list.
    func label(forClassId classId: Int) -> String {
        classes.indices.contains(classId) ? classes[classId] : "object"
    }

    /// Logit above which a single-channel mask counts as object: logit(maskThreshold).
    var logitThreshold: Float {
        let p = min(max(maskThreshold, 1e-6), 1 - 1e-6)
        return Float(log(p / (1 - p)))
    }
}

/// Raw result of one model run, at the model's resolution. Row-major arrays, `width * height` long.
struct ModelOutput: Equatable {
    /// Class id per pixel; 0 is background.
    var classMap: [UInt8]
    /// Depth in metres per pixel, or nil if the model has no depth output.
    var depth: [Float]?
    var width: Int
    var height: Int
}

/// Anything that can turn a resized frame plus meta values into a mask and (optionally) depth.
protocol SegmentationModel {
    var descriptor: ModelDescriptor { get }
    /// `image` is 32BGRA at exactly the descriptor's input size; `meta` follows `descriptor.metaFeatures`.
    func predict(image: CVPixelBuffer, meta: [Float]) throws -> ModelOutput
}
