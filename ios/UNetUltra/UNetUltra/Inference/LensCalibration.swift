import Foundation

/// The per-device lens-position -> focus-distance curve written by `02_train.ipynb`
/// (`calibration.json`). Foundation-only.
///
/// `lensPosition` (0 nearest to 1 furthest) is not a distance; the model only ever sees the
/// calibrated distance this curve gives.
struct LensCalibration: Codable, Equatable {
    var formatVersion: Int
    var kind: String
    var interpolation: String
    var deviceModel: String
    /// Knots, ascending.
    var lensPosition: [Double]
    /// Distance in metres at each knot.
    var distanceM: [Double]
    var nSamples: Int
    var rmseM: Double?
    var createdAt: String

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case kind, interpolation
        case deviceModel = "device_model"
        case lensPosition = "lens_position"
        case distanceM = "distance_m"
        case nSamples = "n_samples"
        case rmseM = "rmse_m"
        case createdAt = "created_at"
    }

    init(lensPosition: [Double], distanceM: [Double], deviceModel: String = "", nSamples: Int = 0,
         rmseM: Double? = nil, createdAt: String = "") {
        self.formatVersion = 1
        self.kind = "lens_position_to_distance"
        self.interpolation = "linear_in_inverse_distance"
        self.deviceModel = deviceModel
        self.lensPosition = lensPosition
        self.distanceM = distanceM
        self.nSamples = nSamples
        self.rmseM = rmseM
        self.createdAt = createdAt
    }

    /// Only the two knot arrays are essential; the rest is descriptive and may be missing.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        formatVersion = try container.decodeIfPresent(Int.self, forKey: .formatVersion) ?? 1
        kind = try container.decodeIfPresent(String.self, forKey: .kind) ?? "lens_position_to_distance"
        interpolation = try container.decodeIfPresent(String.self, forKey: .interpolation)
            ?? "linear_in_inverse_distance"
        deviceModel = try container.decodeIfPresent(String.self, forKey: .deviceModel) ?? ""
        lensPosition = try container.decode([Double].self, forKey: .lensPosition)
        distanceM = try container.decode([Double].self, forKey: .distanceM)
        nSamples = try container.decodeIfPresent(Int.self, forKey: .nSamples) ?? 0
        rmseM = try container.decodeIfPresent(Double.self, forKey: .rmseM)
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ""
    }

    /// Why this calibration cannot be used, or nil if it is fine.
    var problem: String? {
        if formatVersion != 1 { return "calibration format_version \(formatVersion) is not supported" }
        if interpolation != "linear_in_inverse_distance" { return "unknown interpolation \"\(interpolation)\"" }
        if lensPosition.isEmpty || lensPosition.count != distanceM.count {
            return "lens_position and distance_m must be non-empty and the same length"
        }
        if distanceM.contains(where: { !($0 > 0) || !$0.isFinite }) { return "distance_m must be positive" }
        if zip(lensPosition, lensPosition.dropFirst()).contains(where: { $0 > $1 }) {
            return "lens_position must be ascending"
        }
        return nil
    }

    /// Calibrated focus distance in metres for a lens reading: linear in 1/distance between
    /// knots, clamped to the end values outside them (numpy.interp on 1/d, then inverted).
    func focusDistance(lensPosition reading: Double) -> Double? {
        guard problem == nil, let first = lensPosition.first, let last = lensPosition.last else { return nil }
        if reading <= first { return distanceM[0] }
        if reading >= last { return distanceM[distanceM.count - 1] }
        var upper = 1
        while upper < lensPosition.count - 1, lensPosition[upper] < reading { upper += 1 }
        let x0 = lensPosition[upper - 1], x1 = lensPosition[upper]
        let inverse0 = 1 / distanceM[upper - 1], inverse1 = 1 / distanceM[upper]
        let fraction = x1 > x0 ? (reading - x0) / (x1 - x0) : 0
        return 1 / (inverse0 + fraction * (inverse1 - inverse0))
    }
}
