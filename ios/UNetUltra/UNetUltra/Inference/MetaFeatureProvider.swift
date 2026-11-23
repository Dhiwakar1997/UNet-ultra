import Foundation

/// The meta features the app knows how to compute. A model lists the ones it wants by name.
enum MetaFeatureName {
    static let logFocusDistance = "log_focus_distance_m"
    static let fxNorm = "fx_norm"
    static let lensPositionRaw = "lens_position_raw"
}

/// Camera readings a meta vector is computed from. None of them is depth.
struct MetaFeatureInputs: Equatable {
    /// Calibrated focus distance in metres; nil when no calibration is loaded.
    var focusDistanceM: Double?
    /// fx in pixels of the model input, i.e. after the centre-square crop and resize
    /// (fx of the image · input width / crop side).
    var fxModel: Double
    /// Raw `lensPosition`, 0 to 1.
    var lensPosition: Double
}

enum MetaFeatureError: Error, LocalizedError, Equatable {
    case unknownFeature(String)
    case missingCalibration(String)

    var errorDescription: String? {
        switch self {
        case .unknownFeature(let name):
            return "The model asks for the meta feature \"\(name)\", which this app version cannot compute."
        case .missingCalibration(let name):
            return "The meta feature \"\(name)\" needs calibration.json, and none is loaded."
        }
    }
}

/// Computes named meta features in the order a model lists them. Foundation-only.
enum MetaFeatureProvider {
    static func values(for names: [String], inputs: MetaFeatureInputs) throws -> [Float] {
        try names.map { name in
            switch name {
            case MetaFeatureName.logFocusDistance:
                guard let distance = inputs.focusDistanceM, distance > 0 else {
                    throw MetaFeatureError.missingCalibration(name)
                }
                return Float(log(distance))
            case MetaFeatureName.fxNorm:
                return Float(inputs.fxModel / 1000 - 1)
            case MetaFeatureName.lensPositionRaw:
                return Float(inputs.lensPosition)
            default:
                throw MetaFeatureError.unknownFeature(name)
            }
        }
    }
}
