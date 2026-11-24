import CoreML
import Foundation

/// Where things live inside the model library.
enum ModelStoreLayout {
    static let compiledName = "model.mlmodelc"
    static let recordName = "record.json"
    static let bundledPrefix = "bundled:"

    static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("UNetUltra", isDirectory: true)
    }

    /// `Application Support/UNetUltra/Models/<id>/` holds one imported model.
    static var libraryDirectory: URL { supportDirectory.appendingPathComponent("Models", isDirectory: true) }

    /// A file-system-safe folder name.
    static func safeID(_ text: String) -> String {
        let cleaned = text.map { character -> Character in
            character.isLetter || character.isNumber || character == "-" || character == "_" || character == "." ? character : "_"
        }
        let id = String(cleaned).trimmingCharacters(in: CharacterSet(charactersIn: "."))
        return id.isEmpty ? "model" : id
    }
}

/// What the app remembers about an imported model besides its card.
struct ModelRecord: Codable, Equatable {
    var importedAt: Date?
    var selfTest: SelfTestResult?
    /// A name the user typed in the library, shown instead of the card's display name.
    var displayNameOverride: String?
    /// The user chose to use the model although its self-test failed.
    var confirmedDespiteFailure: Bool?
    /// Name for a model imported without a card (a bare `.mlpackage`).
    var legacyName: String?
}

/// One model in the library: bundled with the app, or imported.
struct ModelEntry: Identifiable, Equatable {
    let id: String
    var isBundled: Bool
    /// Nil for a model that came without a card; its description is then read from the Core ML file.
    var card: ModelCard?
    var record: ModelRecord
    var compiledURL: URL
    /// Library folder; nil for a bundled model.
    var folder: URL?

    var displayName: String {
        record.displayNameOverride ?? card?.displayName ?? card?.name ?? record.legacyName ?? id
    }

    var version: String { card?.version ?? "" }
    var architecture: String { card?.architecture ?? "" }

    /// A failed self-test blocks the model until the user confirms they want it anyway.
    var needsConfirmation: Bool {
        record.selfTest?.passed == false && record.confirmedDespiteFailure != true
    }
}

/// The model library: lists bundled and imported models, imports `.unetmodel` bundles, and
/// keeps the active model (and the one before it) loaded for the Inference tab.
///
/// Lens calibration is per device and deliberately not handled here; see `ModelAssets`.
@MainActor
final class ModelStore: ObservableObject {
    @Published private(set) var entries: [ModelEntry] = []
    @Published private(set) var activeID: String?
    @Published private(set) var activeModel: CoreMLSegmentationModel?
    @Published private(set) var isLoadingModel = false
    @Published private(set) var isImporting = false
    /// What the import is doing right now, for the progress indicator.
    @Published private(set) var importStep: String?
    @Published private(set) var computeUnits: ComputeUnitsOption
    /// Latency of the active model on the current compute units; reset when either changes.
    @Published private(set) var latency = LatencyStats()
    /// Set when the user picked a model whose self-test failed; the UI asks for confirmation.
    @Published var pendingConfirmationID: String?
    @Published var message: String?

    static let missingModelHint = "No model. Import a .unetmodel bundle in the Models tab, or run 04_export_coreml "
        + "to copy one into ios/UNetUltra/UNetUltra/Resources/ and rebuild."

    private static let activeKey = "modelStore.activeID"
    private static let unitsKey = "modelStore.computeUnits"
    private static let cacheSize = 2

    private var loaded: [(key: String, model: CoreMLSegmentationModel)] = []
    private var didLoad = false
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        computeUnits = ComputeUnitsOption(rawValue: defaults.string(forKey: Self.unitsKey) ?? "") ?? .all
    }

    var activeEntry: ModelEntry? { entries.first { $0.id == activeID } }

    func entry(_ id: String) -> ModelEntry? { entries.first { $0.id == id } }

    // MARK: Listing

    /// Scan the library and load the last active model, once. Safe to call from every appearance.
    func loadIfNeeded() async {
        guard !didLoad else { return }
        didLoad = true
        Self.migrateLegacyImport()
        refresh()
        let remembered = defaults.string(forKey: Self.activeKey)
        let usable = entries.filter { !$0.needsConfirmation }
        if let target = entries.first(where: { $0.id == remembered && !$0.needsConfirmation })
            ?? usable.first(where: { !$0.isBundled }) ?? usable.first {
            await activate(target.id)
        }
    }

    /// Re-read the library from disk.
    func refresh() {
        entries = Self.bundledEntries() + Self.importedEntries()
    }

    private static func bundledEntries() -> [ModelEntry] {
        let urls = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? []
        return urls.sorted { $0.lastPathComponent < $1.lastPathComponent }.map { url in
            let name = url.deletingPathExtension().lastPathComponent
            return ModelEntry(id: ModelStoreLayout.bundledPrefix + name, isBundled: true, card: nil,
                              record: ModelRecord(legacyName: name), compiledURL: url, folder: nil)
        }
    }

    private static func importedEntries() -> [ModelEntry] {
        let manager = FileManager.default
        let folders = (try? manager.contentsOfDirectory(at: ModelStoreLayout.libraryDirectory,
                                                        includingPropertiesForKeys: nil)) ?? []
        var found: [ModelEntry] = []
        for folder in folders {
            let compiled = folder.appendingPathComponent(ModelStoreLayout.compiledName, isDirectory: true)
            guard manager.fileExists(atPath: compiled.path) else { continue }
            let card = (try? Data(contentsOf: folder.appendingPathComponent(ModelContract.cardFileName)))
                .flatMap { try? ModelCard.decode($0) }
            let record = readRecord(in: folder) ?? ModelRecord(legacyName: folder.lastPathComponent)
            found.append(ModelEntry(id: folder.lastPathComponent, isBundled: false, card: card, record: record,
                                    compiledURL: compiled, folder: folder))
        }
        // Newest import first.
        return found.sorted { ($0.record.importedAt ?? .distantPast) > ($1.record.importedAt ?? .distantPast) }
    }

    nonisolated private static func readRecord(in folder: URL) -> ModelRecord? {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent(ModelStoreLayout.recordName)) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ModelRecord.self, from: data)
    }

    nonisolated private static func writeRecord(_ record: ModelRecord, in folder: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: folder.appendingPathComponent(ModelStoreLayout.recordName), options: .atomic)
    }

    /// The single imported model of earlier app versions becomes an ordinary library entry.
    private static func migrateLegacyImport() {
        let manager = FileManager.default
        let old = ModelStoreLayout.supportDirectory.appendingPathComponent("Imported.mlmodelc", isDirectory: true)
        guard manager.fileExists(atPath: old.path) else { return }
        let nameURL = ModelStoreLayout.supportDirectory.appendingPathComponent("imported_model_name.txt")
        let name = (try? String(contentsOf: nameURL, encoding: .utf8)) ?? "Imported model"
        let folder = ModelStoreLayout.libraryDirectory.appendingPathComponent(ModelStoreLayout.safeID(name), isDirectory: true)
        do {
            try manager.createDirectory(at: folder, withIntermediateDirectories: true)
            let target = folder.appendingPathComponent(ModelStoreLayout.compiledName, isDirectory: true)
            if manager.fileExists(atPath: target.path) { try manager.removeItem(at: target) }
            try manager.moveItem(at: old, to: target)
            try writeRecord(ModelRecord(importedAt: Date(), legacyName: name), in: folder)
            try? manager.removeItem(at: nameURL)
        } catch {
            // Leave the old files where they are; the model can simply be imported again.
        }
    }

    // MARK: Active model

    private static func cacheKey(_ id: String, _ units: ComputeUnitsOption) -> String { "\(id)|\(units.rawValue)" }

    /// Make a model the active one. A model whose self-test failed is not loaded until the
    /// user confirms (`confirmFailedSelfTest`).
    func activate(_ id: String, confirmFailedSelfTest: Bool = false) async {
        guard var entry = entry(id) else { return }
        if entry.needsConfirmation {
            guard confirmFailedSelfTest, let folder = entry.folder else {
                pendingConfirmationID = id
                return
            }
            entry.record.confirmedDespiteFailure = true
            try? Self.writeRecord(entry.record, in: folder)
            refresh()
        }
        pendingConfirmationID = nil
        let key = Self.cacheKey(id, computeUnits)
        if let index = loaded.firstIndex(where: { $0.key == key }) {
            // Still in memory: switching back is instant.
            let hit = loaded.remove(at: index)
            loaded.insert(hit, at: 0)
            setActive(id: id, model: hit.model)
            return
        }
        isLoadingModel = true
        defer { isLoadingModel = false }
        let units = computeUnits
        let snapshot = entry
        do {
            let model = try await Task.detached(priority: .userInitiated) {
                try Self.load(snapshot, units: units)
            }.value
            try CenterSquareCrop.validate(name: model.descriptor.crop)
            loaded.insert((key, model), at: 0)
            if loaded.count > Self.cacheSize { loaded.removeLast(loaded.count - Self.cacheSize) }
            setActive(id: id, model: model)
            message = nil
        } catch {
            message = "\(entry.displayName) could not be loaded: \(error.localizedDescription)"
        }
    }

    private func setActive(id: String, model: CoreMLSegmentationModel) {
        activeID = id
        activeModel = model
        latency = LatencyStats()
        defaults.set(id, forKey: Self.activeKey)
    }

    nonisolated private static func load(_ entry: ModelEntry, units: ComputeUnitsOption) throws -> CoreMLSegmentationModel {
        if let card = entry.card {
            return try CoreMLSegmentationModel(
                compiledModelURL: entry.compiledURL, name: card.name, computeUnits: units.coreML,
                descriptor: card.descriptor(displayNameOverride: entry.record.displayNameOverride))
        }
        // No card: the description is read from the Core ML model itself.
        return try CoreMLSegmentationModel(compiledModelURL: entry.compiledURL, name: entry.displayName,
                                           computeUnits: units.coreML)
    }

    /// Change where Core ML may run models, and reload the active one that way.
    func setComputeUnits(_ units: ComputeUnitsOption) async {
        guard units != computeUnits else { return }
        computeUnits = units
        defaults.set(units.rawValue, forKey: Self.unitsKey)
        latency = LatencyStats()
        if let activeID { await activate(activeID) }
    }

    func recordLatency(_ milliseconds: Double) {
        latency.add(milliseconds)
    }

    private func evict(_ id: String) {
        loaded.removeAll { $0.key.hasPrefix("\(id)|") }
    }

    // MARK: Library edits

    func delete(_ id: String) async {
        guard let entry = entry(id), let folder = entry.folder else { return }
        evict(id)
        do {
            try FileManager.default.removeItem(at: folder)
        } catch {
            message = "\(entry.displayName) could not be deleted: \(error.localizedDescription)"
        }
        refresh()
        if activeID == id {
            activeID = nil
            activeModel = nil
            defaults.removeObject(forKey: Self.activeKey)
            if let next = entries.first(where: { !$0.needsConfirmation }) { await activate(next.id) }
        }
    }

    /// Set (or, with an empty name, clear) the display name shown for an imported model.
    func rename(_ id: String, to newName: String) async {
        guard var entry = entry(id), let folder = entry.folder else { return }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        entry.record.displayNameOverride = trimmed.isEmpty ? nil : trimmed
        do {
            try Self.writeRecord(entry.record, in: folder)
        } catch {
            message = "The name could not be saved: \(error.localizedDescription)"
            return
        }
        evict(id)
        refresh()
        if activeID == id { await activate(id) }
    }

    /// Run a model's self-test again, for example after changing compute units.
    func rerunSelfTest(_ id: String) async {
        guard var entry = entry(id), let folder = entry.folder, let card = entry.card else { return }
        isImporting = true
        importStep = "Running the self-test…"
        defer {
            isImporting = false
            importStep = nil
        }
        let units = computeUnits
        let result = await Task.detached(priority: .userInitiated) {
            ModelSelfTest.run(folder: folder, card: card, computeUnits: units)
        }.value
        entry.record.selfTest = result
        entry.record.confirmedDespiteFailure = nil
        try? Self.writeRecord(entry.record, in: folder)
        refresh()
        message = "\(entry.displayName): \(result.summary)"
    }

    // MARK: Import

    private struct ImportProblem: LocalizedError {
        let errorDescription: String?
    }

    /// Import a `.unetmodel` bundle: unzip, check the card, compile for this device, store,
    /// then self-test. The model becomes active only if the self-test passes.
    func importBundle(from url: URL) async {
        guard !isImporting else {
            message = "Another import is still running."
            return
        }
        isImporting = true
        importStep = "Copying the bundle…"
        defer {
            isImporting = false
            importStep = nil
        }
        let manager = FileManager.default
        let work = manager.temporaryDirectory.appendingPathComponent("model-import-\(UUID().uuidString)", isDirectory: true)
        defer { try? manager.removeItem(at: work) }
        let units = computeUnits
        let report: @Sendable (String) -> Void = { [weak self] step in
            Task { @MainActor in self?.importStep = step }
        }
        do {
            // Copy while we hold access to the picked file; everything after works on the copy.
            let local = work.appendingPathComponent("bundle.\(ModelContract.bundleExtension)")
            let accessing = url.startAccessingSecurityScopedResource()
            do {
                try manager.createDirectory(at: work, withIntermediateDirectories: true)
                try manager.copyItem(at: url, to: local)
            } catch {
                if accessing { url.stopAccessingSecurityScopedResource() }
                throw error
            }
            if accessing { url.stopAccessingSecurityScopedResource() }

            let imported = try await Task.detached(priority: .userInitiated) {
                try await Self.unpackAndStore(bundle: local, work: work, units: units, report: report)
            }.value
            evict(imported.id)
            refresh()
            let name = entry(imported.id)?.displayName ?? imported.id
            if imported.selfTest.passed {
                await activate(imported.id)
                message = "\(name) imported. \(imported.selfTest.summary)"
            } else {
                message = "\(name) imported, but not switched to. \(imported.selfTest.summary)"
            }
        } catch {
            message = "The model could not be imported: \(error.localizedDescription)"
        }
    }

    nonisolated private static func unpackAndStore(
        bundle: URL, work: URL, units: ComputeUnitsOption, report: @Sendable (String) -> Void
    ) async throws -> (id: String, selfTest: SelfTestResult) {
        let manager = FileManager.default
        report("Unzipping…")
        let unzipped = work.appendingPathComponent("unzipped", isDirectory: true)
        let reader: ZipReader
        do {
            reader = try ZipReader(url: bundle)
        } catch ZipReadError.notAZip {
            throw ImportProblem(errorDescription: "This is not a .unetmodel bundle (it is not a zip archive).")
        }
        try reader.extractAll(to: unzipped)

        report("Checking the model card…")
        let cardURL = unzipped.appendingPathComponent(ModelContract.cardFileName)
        guard let cardData = try? Data(contentsOf: cardURL) else {
            throw ImportProblem(errorDescription: "The bundle has no \(ModelContract.cardFileName).")
        }
        let card = try ModelCard.decode(cardData)
        try card.validate()
        let package = unzipped.appendingPathComponent(ModelContract.packageName, isDirectory: true)
        guard manager.fileExists(atPath: package.path) else {
            throw ImportProblem(errorDescription: "The bundle has no \(ModelContract.packageName).")
        }

        report("Compiling for this device (this can take a while the first time)…")
        let compiled = try await MLModel.compileModel(at: package)

        report("Storing the model…")
        let folder = ModelStoreLayout.libraryDirectory.appendingPathComponent(card.libraryID, isDirectory: true)
        if manager.fileExists(atPath: folder.path) { try manager.removeItem(at: folder) }  // same name + version: replace
        try manager.createDirectory(at: folder, withIntermediateDirectories: true)
        try manager.copyItem(at: compiled, to: folder.appendingPathComponent(ModelStoreLayout.compiledName, isDirectory: true))
        try? manager.removeItem(at: compiled)
        try cardData.write(to: folder.appendingPathComponent(ModelContract.cardFileName), options: .atomic)
        let tests = unzipped.appendingPathComponent(ModelContract.selfTestFolder, isDirectory: true)
        if manager.fileExists(atPath: tests.path) {
            try manager.copyItem(at: tests, to: folder.appendingPathComponent(ModelContract.selfTestFolder, isDirectory: true))
        }

        report("Running the self-test…")
        let result = ModelSelfTest.run(folder: folder, card: card, computeUnits: units)
        try writeRecord(ModelRecord(importedAt: Date(), selfTest: result), in: folder)
        return (card.libraryID, result)
    }

    /// Import a bare `.mlpackage` folder (or `.mlmodelc`) with no card: it is compiled and
    /// listed, described by what the Core ML file itself says. There is no self-test for it.
    func importPackage(from url: URL) async {
        guard !isImporting else { return }
        isImporting = true
        importStep = "Compiling the model…"
        defer {
            isImporting = false
            importStep = nil
        }
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let manager = FileManager.default
        let work = manager.temporaryDirectory.appendingPathComponent("model-import-\(UUID().uuidString)", isDirectory: true)
        defer { try? manager.removeItem(at: work) }
        do {
            try manager.createDirectory(at: work, withIntermediateDirectories: true)
            let local = work.appendingPathComponent(url.lastPathComponent, isDirectory: true)
            try manager.copyItem(at: url, to: local)
            let compiled: URL
            if local.pathExtension == "mlmodelc" {
                compiled = local
            } else {
                compiled = try await MLModel.compileModel(at: local)
            }
            let name = url.deletingPathExtension().lastPathComponent
            // Check that it loads and follows the contract before it enters the library.
            let candidate = try await Task.detached(priority: .userInitiated) {
                try CoreMLSegmentationModel(compiledModelURL: compiled, name: name)
            }.value
            try CenterSquareCrop.validate(name: candidate.descriptor.crop)

            let id = ModelStoreLayout.safeID(name)
            let folder = ModelStoreLayout.libraryDirectory.appendingPathComponent(id, isDirectory: true)
            if manager.fileExists(atPath: folder.path) { try manager.removeItem(at: folder) }
            try manager.createDirectory(at: folder, withIntermediateDirectories: true)
            try manager.copyItem(at: compiled, to: folder.appendingPathComponent(ModelStoreLayout.compiledName, isDirectory: true))
            try Self.writeRecord(ModelRecord(importedAt: Date(), legacyName: name), in: folder)
            evict(id)
            refresh()
            await activate(id)
            message = "\(name) imported (no model card, so no self-test)."
        } catch {
            message = "The model could not be imported: \(error.localizedDescription)"
        }
    }
}
