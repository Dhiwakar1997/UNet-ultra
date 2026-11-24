import Foundation

/// The lens calibration the Inference tab runs with, and where it came from.
///
/// Sources, in order of precedence:
/// 1. imported through Files and kept under Application Support,
/// 2. bundled: `calibration.json` dropped into `ios/UNetUltra/UNetUltra/Resources/` before building.
///
/// Calibration belongs to the device, not to a model, so it lives apart from the model
/// library (`ModelStore`), which never touches it.
@MainActor
final class ModelAssets: ObservableObject {
    enum Source: String {
        case imported = "imported"
        case bundled = "bundled"
    }

    @Published private(set) var calibration: LensCalibration?
    @Published private(set) var calibrationSource: Source?
    @Published var message: String?

    static let missingCalibrationHint = "No calibration. Copy calibration.json into ios/UNetUltra/UNetUltra/Resources/ "
        + "and rebuild, or import it from Files."

    private var didLoad = false

    private static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("UNetUltra", isDirectory: true)
    }
    private static var importedCalibrationURL: URL { supportDirectory.appendingPathComponent("calibration.json") }

    var calibrationStatus: String {
        guard let calibration, let calibrationSource else { return Self.missingCalibrationHint }
        let device = calibration.deviceModel.isEmpty ? "unknown device" : calibration.deviceModel
        return "Calibration (\(calibrationSource.rawValue)) · \(calibration.lensPosition.count) knots · "
            + "\(calibration.nSamples) samples · \(device)"
    }

    /// Load whatever is available, once. Safe to call from every appearance of the tab.
    func loadIfNeeded() async {
        guard !didLoad else { return }
        didLoad = true
        loadCalibration()
    }

    // MARK: Calibration

    private func loadCalibration() {
        let candidates: [(URL?, Source)] = [
            (FileManager.default.fileExists(atPath: Self.importedCalibrationURL.path) ? Self.importedCalibrationURL : nil,
             .imported),
            (Bundle.main.url(forResource: "calibration", withExtension: "json"), .bundled),
        ]
        for (url, source) in candidates {
            guard let url else { continue }
            do {
                calibration = try Self.readCalibration(at: url)
                calibrationSource = source
                return
            } catch {
                message = "The \(source.rawValue) calibration.json could not be used: \(error.localizedDescription)"
            }
        }
    }

    private struct CalibrationProblem: LocalizedError {
        let errorDescription: String?
    }

    private static func readCalibration(at url: URL) throws -> LensCalibration {
        let calibration = try JSONDecoder().decode(LensCalibration.self, from: Data(contentsOf: url))
        if let problem = calibration.problem { throw CalibrationProblem(errorDescription: problem) }
        return calibration
    }

    /// Replace the calibration with a `calibration.json` picked in Files.
    func importCalibration(from url: URL) {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        do {
            let loaded = try Self.readCalibration(at: url)
            try FileManager.default.createDirectory(at: Self.supportDirectory, withIntermediateDirectories: true)
            try JSONEncoder().encode(loaded).write(to: Self.importedCalibrationURL, options: .atomic)
            calibration = loaded
            calibrationSource = .imported
            message = "Calibration imported."
        } catch {
            message = "That file is not a usable calibration.json: \(error.localizedDescription)"
        }
    }
}
