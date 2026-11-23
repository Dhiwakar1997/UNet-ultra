import Foundation

/// Packs one stored session into `session_<sessionId>.zip`, in the layout the notebooks expect:
/// `manifest.json` at the zip root and `samples/<sampleId>/...` beside it.
enum SessionExporter {
    /// Files of a sample that belong in the export; anything else in the folder is left out.
    static let sampleFiles = ["image.jpg", "depth_mm.png", "confidence.png", "meta.json"]

    static func zipName(sessionId: String) -> String { "session_\(sessionId).zip" }

    /// Write the zip into `outputDirectory` (replacing an older export) and return its URL.
    @discardableResult
    static func export(session: SessionFiles, to outputDirectory: URL) throws -> URL {
        let manifest = try session.loadManifest()
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let zipURL = outputDirectory.appendingPathComponent(zipName(sessionId: manifest.sessionId))
        let writer = try ZipWriter(url: zipURL)
        try writer.addFile(at: session.manifestURL, name: SessionFiles.manifestName)
        for sampleId in manifest.sampleIds {
            let directory = session.sampleDirectory(sampleId)
            for file in sampleFiles {
                let url = directory.appendingPathComponent(file)
                if FileManager.default.fileExists(atPath: url.path) {
                    try writer.addFile(at: url, name: "samples/\(sampleId)/\(file)")
                }
            }
        }
        try writer.finish()
        return zipURL
    }
}
