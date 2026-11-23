import XCTest
@testable import UNetUltra

/// Writes a tiny session to disk, exports it, and reads the zip back byte by byte.
final class SessionExportTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("unetultra-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func meta(_ sampleId: String, objects: [ObjectAnnotation]) -> SampleMeta {
        SampleMeta(
            sampleId: sampleId, timestamp: "2026-01-01T12:00:00Z", imageSize: [1920, 1440], depthSize: [320, 240],
            intrinsicsImage: Intrinsics(fx: 1450, fy: 1450, cx: 960, cy: 720),
            intrinsicsDepth: Intrinsics(fx: 241.7, fy: 241.7, cx: 160, cy: 120),
            intrinsicsReferenceSize: [4032, 3024], lensPosition: 0.4, focusSettled: true, focusTapIndex: 0,
            deviceOrientation: "portrait", exposure: Exposure(iso: 100, durationS: 0.01),
            backgroundOnly: objects.isEmpty, objects: objects)
    }

    private func makeSession() throws -> SessionFiles {
        let manifest = SessionManifest(
            sessionId: "20260101-120000-abcd", split: .train,
            classes: [SessionFormat.backgroundClass, "airpods_case"],
            deviceModel: "iPhone15,3", iosVersion: "17.5", appVersion: "0.1.0", createdAt: "2026-01-01T12:00:00Z")
        let session = try SessionFiles.create(root: directory.appendingPathComponent("session"), manifest: manifest)
        let object = ObjectAnnotation(classId: 1, label: "airpods_case", tapPointImage: [900, 700],
                                      lidarDistanceAtTapM: 0.3)
        try session.addSample(meta: meta("0001", objects: [object]),
                              imageJPEG: Data([0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9]), depthPNG: Data([9, 8, 7]))
        try session.addSample(meta: meta("0002", objects: []), imageJPEG: Data([0xFF, 0xD8, 0xFF, 0xD9]),
                              depthPNG: Data([6, 5]))
        return session
    }

    // MARK: Session files

    func testAddAndDeleteSamplesKeepTheManifestInStep() throws {
        let session = try makeSession()
        var manifest = try session.loadManifest()
        XCTAssertEqual(manifest.sampleIds, ["0001", "0002"])
        XCTAssertEqual(manifest.maskStatus, ["0001": .pending, "0002": .pending])
        XCTAssertTrue(try session.loadMeta("0002").backgroundOnly)
        XCTAssertEqual(SessionFiles.nextSampleId(after: manifest.sampleIds), "0003")

        manifest = try session.deleteSample("0001")
        XCTAssertEqual(manifest.sampleIds, ["0002"])
        XCTAssertNil(manifest.maskStatus["0001"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: session.sampleDirectory("0001").path))
        XCTAssertEqual(SessionFiles.nextSampleId(after: []), "0001")
    }

    func testSessionIdsAreUniqueAndFileSafe() {
        let first = SessionFiles.makeSessionId(), second = SessionFiles.makeSessionId()
        XCTAssertFalse(first.contains("/") || first.contains(" ") || first.contains(":"))
        XCTAssertEqual(first.count, "20260101-120000-abcd".count)
        XCTAssertEqual(first.prefix(15).count, second.prefix(15).count)
    }

    // MARK: Zip layout

    /// Minimal ZIP reader for the tests: walks the central directory.
    private struct ZipEntry {
        let name: String
        let method: UInt16
        let crc: UInt32
        let compressedSize: UInt32
        let size: UInt32
        let data: Data
    }

    private func readLE<T: FixedWidthInteger>(_ bytes: [UInt8], _ offset: Int, as type: T.Type) -> T {
        var value: T = 0
        for index in 0..<MemoryLayout<T>.size {
            value |= T(bytes[offset + index]) << T(8 * index)
        }
        return value
    }

    private func readZip(_ url: URL) throws -> [ZipEntry] {
        let bytes = [UInt8](try Data(contentsOf: url))
        // No archive comment is written, so the end record is the last 22 bytes.
        let end = bytes.count - 22
        XCTAssertEqual(readLE(bytes, end, as: UInt32.self), 0x0605_4B50)
        let count = Int(readLE(bytes, end + 10, as: UInt16.self))
        XCTAssertEqual(Int(readLE(bytes, end + 8, as: UInt16.self)), count)
        let directorySize = Int(readLE(bytes, end + 12, as: UInt32.self))
        var cursor = Int(readLE(bytes, end + 16, as: UInt32.self))
        XCTAssertEqual(cursor + directorySize, end, "central directory must end where the end record starts")

        var entries: [ZipEntry] = []
        for _ in 0..<count {
            XCTAssertEqual(readLE(bytes, cursor, as: UInt32.self), 0x0201_4B50)
            let method = readLE(bytes, cursor + 10, as: UInt16.self)
            let crc = readLE(bytes, cursor + 16, as: UInt32.self)
            let compressedSize = readLE(bytes, cursor + 20, as: UInt32.self)
            let size = readLE(bytes, cursor + 24, as: UInt32.self)
            let nameLength = Int(readLE(bytes, cursor + 28, as: UInt16.self))
            let extraLength = Int(readLE(bytes, cursor + 30, as: UInt16.self))
            let commentLength = Int(readLE(bytes, cursor + 32, as: UInt16.self))
            let localOffset = Int(readLE(bytes, cursor + 42, as: UInt32.self))
            let name = String(decoding: bytes[(cursor + 46)..<(cursor + 46 + nameLength)], as: UTF8.self)

            // The local header must repeat the same facts and sit right before the data.
            XCTAssertEqual(readLE(bytes, localOffset, as: UInt32.self), 0x0403_4B50)
            XCTAssertEqual(readLE(bytes, localOffset + 14, as: UInt32.self), crc)
            XCTAssertEqual(readLE(bytes, localOffset + 18, as: UInt32.self), compressedSize)
            XCTAssertEqual(readLE(bytes, localOffset + 22, as: UInt32.self), size)
            let localNameLength = Int(readLE(bytes, localOffset + 26, as: UInt16.self))
            let localExtraLength = Int(readLE(bytes, localOffset + 28, as: UInt16.self))
            XCTAssertEqual(localNameLength, nameLength)
            let localName = String(decoding: bytes[(localOffset + 30)..<(localOffset + 30 + localNameLength)],
                                   as: UTF8.self)
            XCTAssertEqual(localName, name)
            let dataStart = localOffset + 30 + localNameLength + localExtraLength
            let data = Data(bytes[dataStart..<(dataStart + Int(size))])

            entries.append(ZipEntry(name: name, method: method, crc: crc, compressedSize: compressedSize,
                                    size: size, data: data))
            cursor += 46 + nameLength + extraLength + commentLength
        }
        return entries
    }

    func testCRC32MatchesKnownValues() {
        XCTAssertEqual(CRC32.checksum(Data()), 0)
        XCTAssertEqual(CRC32.checksum(Data("123456789".utf8)), 0xCBF4_3926)
        XCTAssertEqual(CRC32.checksum(Data("The quick brown fox jumps over the lazy dog".utf8)), 0x414F_A339)
    }

    func testExportedZipHasTheContractLayout() throws {
        let session = try makeSession()
        // A stray file in a sample folder must not leak into the export.
        try Data([1]).write(to: session.sampleDirectory("0001").appendingPathComponent("thumbnail.tmp"))

        let zipURL = try SessionExporter.export(session: session, to: directory.appendingPathComponent("out"))
        XCTAssertEqual(zipURL.lastPathComponent, "session_20260101-120000-abcd.zip")

        let entries = try readZip(zipURL)
        XCTAssertEqual(entries.map(\.name), [
            "manifest.json",
            "samples/0001/image.jpg", "samples/0001/depth_mm.png", "samples/0001/meta.json",
            "samples/0002/image.jpg", "samples/0002/depth_mm.png", "samples/0002/meta.json",
        ])
        for entry in entries {
            XCTAssertEqual(entry.method, 0, "\(entry.name) should be stored")
            XCTAssertEqual(entry.compressedSize, entry.size)
            XCTAssertEqual(entry.crc, CRC32.checksum(entry.data), "CRC of \(entry.name)")
            XCTAssertFalse(entry.name.hasPrefix("/") || entry.name.contains("\\"))
        }
        XCTAssertEqual(entries[1].data, Data([0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9]))
        XCTAssertEqual(entries[5].data, Data([6, 5]))

        // The files inside decode back into the models the app wrote.
        let manifest = try SessionJSON.decoder().decode(SessionManifest.self, from: entries[0].data)
        XCTAssertEqual(manifest.sampleIds, ["0001", "0002"])
        let meta = try SessionJSON.decoder().decode(SampleMeta.self, from: entries[3].data)
        XCTAssertEqual(meta.objects.first?.tapPointImage, [900, 700])
    }

    func testExportReplacesAnOlderZipAndSkipsDeletedSamples() throws {
        let session = try makeSession()
        let output = directory.appendingPathComponent("out")
        try SessionExporter.export(session: session, to: output)
        try session.deleteSample("0001")
        let entries = try readZip(try SessionExporter.export(session: session, to: output))
        XCTAssertEqual(entries.map(\.name), [
            "manifest.json", "samples/0002/image.jpg", "samples/0002/depth_mm.png", "samples/0002/meta.json",
        ])
    }

    func testZipWriterRejectsBadNames() throws {
        let writer = try ZipWriter(url: directory.appendingPathComponent("bad.zip"))
        XCTAssertThrowsError(try writer.addEntry(name: "/absolute.txt", data: Data()))
        XCTAssertThrowsError(try writer.addEntry(name: "back\\slash.txt", data: Data()))
        XCTAssertThrowsError(try writer.addEntry(name: "", data: Data()))
        try writer.finish()
    }
}
