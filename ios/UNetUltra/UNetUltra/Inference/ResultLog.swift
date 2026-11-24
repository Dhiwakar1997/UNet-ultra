import Foundation

/// CSV text helpers. Foundation-only.
enum CSV {
    /// One field, quoted when it holds a comma, a quote or a line break; quotes are doubled.
    static func escape(_ field: String) -> String {
        let needsQuotes = field.contains(",") || field.contains("\"") || field.contains("\n") || field.contains("\r")
        guard needsQuotes else { return field }
        return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    static func line(_ fields: [String]) -> String {
        fields.map(escape).joined(separator: ",")
    }
}

/// An append-only CSV file with a fixed header.
struct ResultLog {
    let fileURL: URL
    let header: [String]

    /// The log file to use in `directory`: `<baseName>.csv`, unless a file of that name already
    /// exists with a different header (an older column set). Then `<baseName>_v2.csv`, `_v3` and
    /// so on, so rows with different columns are never mixed in one file.
    static func resolve(directory: URL, baseName: String, header: [String]) -> ResultLog {
        let headerLine = CSV.line(header)
        var version = 1
        while true {
            let name = version == 1 ? "\(baseName).csv" : "\(baseName)_v\(version).csv"
            let url = directory.appendingPathComponent(name)
            guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else {
                return ResultLog(fileURL: url, header: header)
            }
            let firstLine = text.split(whereSeparator: { $0 == "\n" || $0 == "\r\n" }).first.map { String($0) } ?? ""
            if firstLine == headerLine || version >= 50 {
                return ResultLog(fileURL: url, header: header)
            }
            version += 1
        }
    }

    /// Append one row, creating the file with its header first if needed.
    func append(_ fields: [String]) throws {
        let manager = FileManager.default
        if !manager.fileExists(atPath: fileURL.path) {
            try manager.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data((CSV.line(header) + "\n").utf8).write(to: fileURL, options: .atomic)
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data((CSV.line(fields) + "\n").utf8))
    }

    /// Number of data rows (records, not text lines: quoted line breaks do not count).
    func rowCount() -> Int {
        guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { return 0 }
        var records = 0, inQuotes = false, lineHasContent = false
        for character in text {
            if character == "\"" { inQuotes.toggle() }
            if (character == "\n" || character == "\r\n") && !inQuotes {
                if lineHasContent { records += 1 }
                lineHasContent = false
            } else {
                lineHasContent = true
            }
        }
        if lineHasContent { records += 1 }
        return max(records - 1, 0)
    }
}

/// One logged inference: predictions, the LiDAR reference and optional hand measurements.
struct InferenceLogRow {
    static let header = [
        "timestamp", "model_name", "device_model", "lens_position", "focus_distance_cm",
        "pred_distance_cm", "pred_width_cm", "pred_height_cm", "lidar_distance_cm", "distance_diff_cm",
        "mask_area_px", "latency_ms", "class_label", "measured_width_cm", "measured_height_cm",
        "measured_distance_cm", "notes", "model_version", "architecture", "compute_units",
    ]

    var timestamp: String
    var deviceModel: String
    var result: InferenceResult
    var measuredWidthCm: Double?
    var measuredHeightCm: Double?
    var measuredDistanceCm: Double?
    var notes: String

    private static func number(_ value: Double?, scale: Double = 1, digits: Int = 2) -> String {
        guard let value, value.isFinite else { return "" }
        return String(format: "%.\(digits)f", value * scale)
    }

    var fields: [String] {
        [
            timestamp, result.modelName, deviceModel,
            Self.number(result.lensPosition, digits: 4),
            Self.number(result.focusDistanceM, scale: 100),
            Self.number(result.measurement?.distanceM, scale: 100),
            Self.number(result.measurement?.widthM, scale: 100),
            Self.number(result.measurement?.heightM, scale: 100),
            Self.number(result.lidarDistanceM, scale: 100),
            Self.number(result.distanceDifferenceM, scale: 100),
            String(result.areaPx),
            Self.number(result.latencyMs, digits: 1),
            result.classLabel,
            Self.number(measuredWidthCm), Self.number(measuredHeightCm), Self.number(measuredDistanceCm),
            notes, result.modelVersion, result.architecture, result.computeUnits,
        ]
    }
}
