import CoreML
import Foundation

/// The model and the lens calibration the Inference tab runs with, and where they came from.
///
/// Sources, in order of precedence:
/// 1. imported through Files and kept under Application Support,
/// 2. bundled: `UNetUltra.mlpackage` and `calibration.json` dropped into
///    `ios/UNetUltra/UNetUltra/Resources/` before building.
///
/// Calibration belongs to the device, not to a model, so the two are loaded independently.
@MainActor
final class ModelAssets: ObservableObject {
    enum Source: String {
        case imported = "imported"
        case bundled = "bundled"
    }

    @Published private(set) var model: CoreMLSegmentationModel?
    @Published private(set) var modelSource: Source?
    @Published private(set) var calibration: LensCalibration?
    @Published private(set) var calibrationSource: Source?
    /// True while a model is being copied, compiled or loaded.
    @Published private(set) var isLoadingModel = false
    @Published var message: String?

    static let bundledModelName = "UNetUltra"
    static let missingModelHint = "No model. Copy UNetUltra.mlpackage into ios/UNetUltra/UNetUltra/Resources/ "
        + "and rebuild, or import the .mlpackage folder from Files."
    static let missingCalibrationHint = "No calibration. Copy calibration.json into ios/UNetUltra/UNetUltra/Resources/ "
        + "and rebuild, or import it from Files."

    private var didLoad = false

    private static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("UNetUltra", isDirectory: true)
    }
    private static var importedModelURL: URL { supportDirectory.appendingPathComponent("Imported.mlmodelc", isDirectory: true) }
    private static var importedModelNameURL: URL { supportDirectory.appendingPathComponent("imported_model_name.txt") }
    private static var importedCalibrationURL: URL { supportDirectory.appendingPathComponent("calibration.json") }

    var modelStatus: String {
        if isLoadingModel { return "Loading model…" }
        guard let model, let modelSource else { return Self.missingModelHint }
        let d = model.descriptor
        return "\(d.displayName) (\(modelSource.rawValue)) · \(d.inputWidth)×\(d.inputHeight) · "
            + (d.metaFeatures.isEmpty ? "image only" : d.metaFeatures.joined(separator: ", "))
            + (d.hasDepth ? "" : " · no depth output")
    }

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
        await loadModel()
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

    // MARK: Model

    private func loadModel() async {
        isLoadingModel = true
        defer { isLoadingModel = false }
        let manager = FileManager.default
        if manager.fileExists(atPath: Self.importedModelURL.path) {
            let name = (try? String(contentsOf: Self.importedModelNameURL, encoding: .utf8)) ?? "Imported model"
            if await load(compiledModelURL: Self.importedModelURL, name: name, source: .imported) { return }
        }
        let bundled = Bundle.main.url(forResource: Self.bundledModelName, withExtension: "mlmodelc")
            ?? Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)?.first
        if let bundled {
            _ = await load(compiledModelURL: bundled, name: bundled.deletingPathExtension().lastPathComponent,
                           source: .bundled)
        }
    }

    private func load(compiledModelURL: URL, name: String, source: Source) async -> Bool {
        do {
            let loaded = try await Task.detached(priority: .userInitiated) {
                try CoreMLSegmentationModel(compiledModelURL: compiledModelURL, name: name)
            }.value
            try CenterSquareCrop.validate(name: loaded.descriptor.crop)
            model = loaded
            modelSource = source
            return true
        } catch {
            message = "The \(source.rawValue) model could not be loaded: \(error.localizedDescription)"
            return false
        }
    }

    /// Import a `.mlpackage` folder (or an already compiled `.mlmodelc`) picked in Files:
    /// copy it in, compile it on the device, keep the compiled model for the next launch.
    func importModel(from url: URL) async {
        isLoadingModel = true
        defer { isLoadingModel = false }
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let manager = FileManager.default
        let work = manager.temporaryDirectory.appendingPathComponent("model-import-\(UUID().uuidString)", isDirectory: true)
        defer { try? manager.removeItem(at: work) }
        do {
            try manager.createDirectory(at: work, withIntermediateDirectories: true)
            let local = work.appendingPathComponent(url.lastPathComponent, isDirectory: true)
            try manager.copyItem(at: url, to: local)
            // Compiling can take several seconds for a new model; it runs off the main actor.
            let compiled: URL
            if local.pathExtension == "mlmodelc" {
                compiled = local
            } else {
                compiled = try await MLModel.compileModel(at: local)
            }
            let name = url.deletingPathExtension().lastPathComponent

            // Check that it loads and follows the contract before replacing the stored model.
            let staged = work.appendingPathComponent("Staged.mlmodelc", isDirectory: true)
            try manager.copyItem(at: compiled, to: staged)
            let candidate = try await Task.detached(priority: .userInitiated) {
                try CoreMLSegmentationModel(compiledModelURL: staged, name: name)
            }.value
            try CenterSquareCrop.validate(name: candidate.descriptor.crop)

            try manager.createDirectory(at: Self.supportDirectory, withIntermediateDirectories: true)
            if manager.fileExists(atPath: Self.importedModelURL.path) {
                try manager.removeItem(at: Self.importedModelURL)
            }
            try manager.copyItem(at: compiled, to: Self.importedModelURL)
            try Data(name.utf8).write(to: Self.importedModelNameURL, options: .atomic)
            if await load(compiledModelURL: Self.importedModelURL, name: name, source: .imported) {
                message = "Model \(name) imported."
            }
        } catch {
            message = "The model could not be imported: \(error.localizedDescription)"
        }
    }
}
