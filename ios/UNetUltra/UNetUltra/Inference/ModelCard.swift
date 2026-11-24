import Foundation

/// The model contract this app understands. A bundle whose card names another major version
/// is refused rather than run with the wrong assumptions.
enum ModelContract {
    static let supportedMajorVersion = 1
    static let cardFileName = "model_card.json"
    static let packageName = "model.mlpackage"
    static let selfTestFolder = "self_test"
    static let bundleExtension = "unetmodel"
}

enum ModelCardError: Error, LocalizedError, Equatable {
    case unsupportedContract(found: String)
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedContract(let found):
            return "This model follows model contract version \(found), but this app only understands version "
                + "\(ModelContract.supportedMajorVersion). Update the app, or export the model again for version "
                + "\(ModelContract.supportedMajorVersion)."
        case .invalid(let reason):
            return "The model card is not usable: \(reason)"
        }
    }
}

/// `contract_version` as written in the card: an integer, a number or a string such as "1.2".
struct ContractVersion: Codable, Equatable {
    var text: String

    init(_ text: String) { self.text = text }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let integer = try? container.decode(Int.self) {
            text = String(integer)
        } else if let number = try? container.decode(Double.self) {
            text = String(number)
        } else {
            text = try container.decode(String.self)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let integer = Int(text) {
            try container.encode(integer)
        } else {
            try container.encode(text)
        }
    }

    /// The part before the first dot, or nil if it is not a number.
    var major: Int? {
        guard let first = text.split(separator: ".", omittingEmptySubsequences: false).first else { return nil }
        return Int(String(first))
    }
}

/// `model_card.json` from a `.unetmodel` bundle. Foundation-only.
///
/// Only `contract_version`, `name` and `input` are required; everything else has a sensible
/// default, and keys this version does not know are ignored.
struct ModelCard: Codable, Equatable {
    struct Input: Codable, Equatable {
        var width: Int
        var height: Int
        var crop: String?
    }

    struct MaskOutput: Codable, Equatable {
        var channels: Int?
    }

    struct DepthOutput: Codable, Equatable {
        var units: String?
    }

    struct Outputs: Codable, Equatable {
        var maskLogits: MaskOutput?
        /// Absent for a model with no depth output.
        var depth: DepthOutput?

        enum CodingKeys: String, CodingKey {
            case maskLogits = "mask_logits"
            case depth
        }
    }

    struct TestMetrics: Codable, Equatable {
        var dice: Double?
        var distanceErrPct: Double?
        var sizeErrPct: Double?

        enum CodingKeys: String, CodingKey {
            case dice
            case distanceErrPct = "distance_err_pct"
            case sizeErrPct = "size_err_pct"
        }
    }

    var contractVersion: ContractVersion
    var name: String
    var displayName: String?
    var version: String?
    var architecture: String?
    var input: Input
    /// Empty for an image-only model, which then has no `meta` input at all.
    var metaFeatures: [String]?
    var outputs: Outputs?
    var classes: [String]?
    var maskThreshold: Double?
    var trainedOnSessions: [String]?
    var testMetrics: TestMetrics?
    var sizeMb: Double?
    var precision: String?
    var notes: String?
    var createdAt: String?
    var sha256: String?

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case name
        case displayName = "display_name"
        case version, architecture, input
        case metaFeatures = "meta_features"
        case outputs, classes
        case maskThreshold = "mask_threshold"
        case trainedOnSessions = "trained_on_sessions"
        case testMetrics = "test_metrics"
        case sizeMb = "size_mb"
        case precision, notes
        case createdAt = "created_at"
        case sha256
    }

    static func decode(_ data: Data) throws -> ModelCard {
        do {
            return try JSONDecoder().decode(ModelCard.self, from: data)
        } catch {
            throw ModelCardError.invalid("model_card.json could not be read (\(Self.describe(error))).")
        }
    }

    private static func describe(_ error: Error) -> String {
        guard let decoding = error as? DecodingError else { return error.localizedDescription }
        switch decoding {
        case .keyNotFound(let key, _): return "missing \"\(key.stringValue)\""
        case .typeMismatch(_, let context), .valueNotFound(_, let context):
            return "wrong value for \"\(context.codingPath.map(\.stringValue).joined(separator: "."))\""
        case .dataCorrupted: return "not valid JSON"
        @unknown default: return "unreadable"
        }
    }

    static let knownMetaFeatures: Set<String> = [
        MetaFeatureName.logFocusDistance, MetaFeatureName.fxNorm, MetaFeatureName.lensPositionRaw,
    ]

    /// Check the card against the contract. Throws the reason the bundle must be refused.
    func validate() throws {
        guard contractVersion.major == ModelContract.supportedMajorVersion else {
            throw ModelCardError.unsupportedContract(found: contractVersion.text)
        }
        guard !name.trimmingCharacters(in: .whitespaces).isEmpty else { throw ModelCardError.invalid("it has no name.") }
        guard input.width > 0, input.height > 0 else {
            throw ModelCardError.invalid("the input size \(input.width)×\(input.height) is not valid.")
        }
        if let crop = input.crop, crop != CenterSquareCrop.name {
            throw ModelCardError.invalid("it asks for the crop \"\(crop)\"; this app only supports \"\(CenterSquareCrop.name)\".")
        }
        if let classes, classes.count < 2 {
            throw ModelCardError.invalid("classes must list background and at least one object class.")
        }
        if let channels = outputs?.maskLogits?.channels, channels < 1 {
            throw ModelCardError.invalid("mask_logits must have at least one channel.")
        }
        if let threshold = maskThreshold, !(threshold > 0 && threshold < 1) {
            throw ModelCardError.invalid("mask_threshold must be between 0 and 1.")
        }
        for feature in metaFeatures ?? [] where !Self.knownMetaFeatures.contains(feature) {
            throw ModelCardError.invalid("it asks for the meta feature \"\(feature)\", which this app cannot compute.")
        }
    }

    var hasDepthOutput: Bool {
        // A card with no `outputs` block at all says nothing either way: read depth if the model has it.
        outputs == nil || outputs?.depth != nil
    }

    /// What the app needs to run the model, taken from the card rather than from the Core ML file.
    func descriptor(displayNameOverride: String? = nil) -> ModelDescriptor {
        let classList = classes ?? ModelDescriptor.defaultClasses
        return ModelDescriptor(
            name: name,
            displayName: displayNameOverride ?? displayName ?? name,
            inputWidth: input.width, inputHeight: input.height,
            metaFeatures: metaFeatures ?? [],
            maskChannels: outputs?.maskLogits?.channels ?? (classList.count <= 2 ? 1 : classList.count),
            hasDepth: hasDepthOutput,
            classes: classList,
            maskThreshold: maskThreshold ?? 0.5,
            crop: input.crop ?? CenterSquareCrop.name)
    }

    /// Folder name for this model in the library: the same name and version always map to the same folder.
    var libraryID: String {
        let raw = version.map { "\(name)-\($0)" } ?? name
        let cleaned = raw.map { character -> Character in
            character.isLetter || character.isNumber || character == "-" || character == "_" || character == "." ? character : "_"
        }
        let id = String(cleaned).trimmingCharacters(in: CharacterSet(charactersIn: "."))
        return id.isEmpty ? "model" : id
    }
}
