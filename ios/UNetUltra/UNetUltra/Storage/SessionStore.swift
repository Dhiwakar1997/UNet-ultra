import Foundation

/// The capture sessions kept on the phone, under `Documents/sessions/<sessionId>/`, and the
/// one currently being captured into. Publishes what the Capture tab and gallery show.
@MainActor
final class SessionStore: ObservableObject {
    @Published private(set) var manifest: SessionManifest?
    /// Metadata of the current session's samples, in capture order.
    @Published private(set) var samples: [SampleMeta] = []
    @Published private(set) var sessionIds: [String] = []
    @Published var errorMessage: String?

    private(set) var files: SessionFiles?
    let sessionsDirectory: URL

    init(sessionsDirectory: URL? = nil) {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.sessionsDirectory = sessionsDirectory ?? documents.appendingPathComponent("sessions", isDirectory: true)
        refreshSessionIds()
    }

    var sampleCount: Int { samples.count }
    var backgroundOnlyCount: Int { samples.filter(\.backgroundOnly).count }

    /// Class names of the current session without "background"; index + 1 is the class id.
    var objectClasses: [String] { Array((manifest?.classes ?? []).dropFirst()) }

    /// Start a new session. `classes` are the object classes; "background" is added as class 0.
    func startSession(classes: [String], split: SessionSplit, deviceModel: String, iosVersion: String,
                      appVersion: String) {
        let names = classes.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let manifest = SessionManifest(
            sessionId: SessionFiles.makeSessionId(), split: split,
            classes: [SessionFormat.backgroundClass] + (names.isEmpty ? ["airpods_case"] : names),
            deviceModel: deviceModel, iosVersion: iosVersion, appVersion: appVersion,
            createdAt: SessionFiles.timestamp())
        do {
            let root = sessionsDirectory.appendingPathComponent(manifest.sessionId, isDirectory: true)
            files = try SessionFiles.create(root: root, manifest: manifest)
            self.manifest = manifest
            samples = []
            refreshSessionIds()
        } catch {
            errorMessage = "Could not create the session: \(error.localizedDescription)"
        }
    }

    /// Reopen a stored session to add to it, review it or export it.
    func openSession(_ sessionId: String) {
        let candidate = SessionFiles(root: sessionsDirectory.appendingPathComponent(sessionId, isDirectory: true))
        do {
            let loaded = try candidate.loadManifest()
            files = candidate
            manifest = loaded
            samples = loaded.sampleIds.compactMap { try? candidate.loadMeta($0) }
        } catch {
            errorMessage = "Could not open session \(sessionId): \(error.localizedDescription)"
        }
    }

    /// The id the next sample will get.
    func nextSampleId() -> String {
        SessionFiles.nextSampleId(after: (manifest?.sampleIds ?? []) + samples.map(\.sampleId))
    }

    func addSample(meta: SampleMeta, imageJPEG: Data, depthPNG: Data) -> Bool {
        guard let files else { return false }
        do {
            manifest = try files.addSample(meta: meta, imageJPEG: imageJPEG, depthPNG: depthPNG)
            samples.append(meta)
            return true
        } catch {
            errorMessage = "Could not save the sample: \(error.localizedDescription)"
            return false
        }
    }

    func deleteSample(_ sampleId: String) {
        guard let files else { return }
        do {
            manifest = try files.deleteSample(sampleId)
            samples.removeAll { $0.sampleId == sampleId }
        } catch {
            errorMessage = "Could not delete the sample: \(error.localizedDescription)"
        }
    }

    func imageURL(_ sampleId: String) -> URL? { files?.imageURL(sampleId) }

    /// Zip the current session into the temporary directory and return the zip's URL.
    func exportCurrentSession() async -> URL? {
        guard let files else { return nil }
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("exports", isDirectory: true)
        do {
            return try await Task.detached(priority: .userInitiated) {
                try SessionExporter.export(session: files, to: output)
            }.value
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
            return nil
        }
    }

    private func refreshSessionIds() {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: sessionsDirectory, includingPropertiesForKeys: nil)) ?? []
        sessionIds = contents
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent(SessionFiles.manifestName).path) }
            .map(\.lastPathComponent)
            .sorted(by: >)
    }
}
