import Foundation

/// One capture session on disk:
///
///     <root>/manifest.json
///     <root>/samples/<sampleId>/{image.jpg, depth_mm.png, meta.json}
///
/// Foundation-only: images arrive already encoded, so this is testable without a camera.
struct SessionFiles: Equatable {
    static let manifestName = "manifest.json"
    static let imageName = "image.jpg"
    static let depthName = "depth_mm.png"
    static let metaName = "meta.json"

    let root: URL

    var manifestURL: URL { root.appendingPathComponent(Self.manifestName) }
    var samplesDirectory: URL { root.appendingPathComponent("samples", isDirectory: true) }

    func sampleDirectory(_ sampleId: String) -> URL {
        samplesDirectory.appendingPathComponent(sampleId, isDirectory: true)
    }

    func imageURL(_ sampleId: String) -> URL { sampleDirectory(sampleId).appendingPathComponent(Self.imageName) }
    func metaURL(_ sampleId: String) -> URL { sampleDirectory(sampleId).appendingPathComponent(Self.metaName) }

    /// Create the session folder and write its manifest.
    static func create(root: URL, manifest: SessionManifest) throws -> SessionFiles {
        let files = SessionFiles(root: root)
        try FileManager.default.createDirectory(at: files.samplesDirectory, withIntermediateDirectories: true)
        try files.saveManifest(manifest)
        return files
    }

    func loadManifest() throws -> SessionManifest {
        try SessionJSON.decoder().decode(SessionManifest.self, from: Data(contentsOf: manifestURL))
    }

    func saveManifest(_ manifest: SessionManifest) throws {
        try SessionJSON.encoder().encode(manifest).write(to: manifestURL, options: .atomic)
    }

    func loadMeta(_ sampleId: String) throws -> SampleMeta {
        try SessionJSON.decoder().decode(SampleMeta.self, from: Data(contentsOf: metaURL(sampleId)))
    }

    /// The next unused sample id: a zero-padded counter one above the highest id ever listed.
    static func nextSampleId(after existing: [String]) -> String {
        let highest = existing.compactMap { Int($0) }.max() ?? 0
        return String(format: "%04d", highest + 1)
    }

    /// Write one sample and list it in the manifest as "pending". Returns the updated manifest.
    @discardableResult
    func addSample(meta: SampleMeta, imageJPEG: Data, depthPNG: Data) throws -> SessionManifest {
        var manifest = try loadManifest()
        let directory = sampleDirectory(meta.sampleId)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try imageJPEG.write(to: directory.appendingPathComponent(Self.imageName), options: .atomic)
        try depthPNG.write(to: directory.appendingPathComponent(Self.depthName), options: .atomic)
        try SessionJSON.encoder().encode(meta).write(to: directory.appendingPathComponent(Self.metaName), options: .atomic)
        if !manifest.sampleIds.contains(meta.sampleId) {
            manifest.sampleIds.append(meta.sampleId)
        }
        manifest.maskStatus[meta.sampleId] = .pending
        try saveManifest(manifest)
        return manifest
    }

    /// Remove one sample from disk and from the manifest. Returns the updated manifest.
    @discardableResult
    func deleteSample(_ sampleId: String) throws -> SessionManifest {
        var manifest = try loadManifest()
        manifest.sampleIds.removeAll { $0 == sampleId }
        manifest.maskStatus[sampleId] = nil
        try saveManifest(manifest)
        let directory = sampleDirectory(sampleId)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
        return manifest
    }

    /// Session ids are a local timestamp plus a short random suffix, e.g. `20260101-120000-a3f9`.
    static func makeSessionId(date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let suffix = String(format: "%04x", Int.random(in: 0...0xFFFF))
        return "\(formatter.string(from: date))-\(suffix)"
    }

    static func timestamp(_ date: Date = Date()) -> String {
        ISO8601DateFormatter().string(from: date)
    }
}
