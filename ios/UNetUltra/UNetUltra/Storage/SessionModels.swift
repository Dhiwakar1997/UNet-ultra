import Foundation

/// Codable mirrors of `manifest.json` and `meta.json`: the data contract with the Python notebooks.
/// Field names and null handling must match `python/unetultra/validate.py`.

enum SessionFormat {
    static let version = 1
    static let backgroundClass = "background"
}

enum SessionSplit: String, Codable, CaseIterable {
    case train, test
}

enum MaskStatus: String, Codable {
    case pending, accepted, rejected
}

struct SessionManifest: Codable, Equatable {
    var formatVersion: Int = SessionFormat.version
    var sessionId: String
    var split: SessionSplit
    /// Class names; index is the class id and index 0 is always "background".
    var classes: [String]
    var deviceModel: String
    var iosVersion: String
    var appVersion: String
    var createdAt: String
    var sampleIds: [String] = []
    var maskStatus: [String: MaskStatus] = [:]

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case sessionId = "session_id"
        case split, classes
        case deviceModel = "device_model"
        case iosVersion = "ios_version"
        case appVersion = "app_version"
        case createdAt = "created_at"
        case sampleIds = "sample_ids"
        case maskStatus = "mask_status"
    }
}

/// Pinhole intrinsics in pixels, valid for one specific image size.
struct Intrinsics: Codable, Equatable {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
}

struct Exposure: Codable, Equatable {
    var iso: Double
    var durationS: Double

    enum CodingKeys: String, CodingKey {
        case iso
        case durationS = "duration_s"
    }
}

/// Size measured by hand with a ruler. Missing values are written as JSON null.
struct TrueSize: Codable, Equatable {
    var width: Double?
    var height: Double?

    init(width: Double? = nil, height: Double? = nil) {
        self.width = width
        self.height = height
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(width, forKey: .width)
        try container.encode(height, forKey: .height)
    }
}

/// A positive or negative click added during mask review on the Mac.
struct ExtraPoint: Codable, Equatable {
    var point: [Double]
    var positive: Bool
}

struct ObjectAnnotation: Codable, Equatable {
    var classId: Int
    var label: String
    /// Tap position in saved-image pixels (sensor orientation), [x, y].
    var tapPointImage: [Double]
    var extraPoints: [ExtraPoint] = []
    var lidarDistanceAtTapM: Double?
    var trueSizeCm = TrueSize()
    var facing: String?
    var trueDistanceCm: Double?

    enum CodingKeys: String, CodingKey {
        case classId = "class_id"
        case label
        case tapPointImage = "tap_point_image"
        case extraPoints = "extra_points"
        case lidarDistanceAtTapM = "lidar_distance_at_tap_m"
        case trueSizeCm = "true_size_cm"
        case facing
        case trueDistanceCm = "true_distance_cm"
    }

    /// Written by hand so that optional values appear as explicit nulls, as the contract shows.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(classId, forKey: .classId)
        try container.encode(label, forKey: .label)
        try container.encode(tapPointImage, forKey: .tapPointImage)
        try container.encode(extraPoints, forKey: .extraPoints)
        try container.encode(lidarDistanceAtTapM, forKey: .lidarDistanceAtTapM)
        try container.encode(trueSizeCm, forKey: .trueSizeCm)
        try container.encode(facing, forKey: .facing)
        try container.encode(trueDistanceCm, forKey: .trueDistanceCm)
    }
}

struct SampleMeta: Codable, Equatable {
    var sampleId: String
    var timestamp: String
    /// [W, H] of image.jpg, in sensor orientation.
    var imageSize: [Int]
    /// [w, h] of depth_mm.png.
    var depthSize: [Int]
    var intrinsicsImage: Intrinsics
    var intrinsicsDepth: Intrinsics
    var intrinsicsReferenceSize: [Int]
    var lensPosition: Double
    var focusSettled: Bool
    var focusTapIndex: Int
    var deviceOrientation: String
    var exposure: Exposure
    var backgroundOnly: Bool
    var objects: [ObjectAnnotation]
    var notes: String = ""

    enum CodingKeys: String, CodingKey {
        case sampleId = "sample_id"
        case timestamp
        case imageSize = "image_size"
        case depthSize = "depth_size"
        case intrinsicsImage = "intrinsics_image"
        case intrinsicsDepth = "intrinsics_depth"
        case intrinsicsReferenceSize = "intrinsics_reference_size"
        case lensPosition = "lens_position"
        case focusSettled = "focus_settled"
        case focusTapIndex = "focus_tap_index"
        case deviceOrientation = "device_orientation"
        case exposure
        case backgroundOnly = "background_only"
        case objects, notes
    }
}

enum SessionJSON {
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    static func decoder() -> JSONDecoder { JSONDecoder() }
}
