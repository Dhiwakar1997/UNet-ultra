import XCTest
@testable import UNetUltra

/// Model library logic that needs no camera and no real model: cards, the contract check,
/// bundle unzipping, the self-test arithmetic, latency statistics and the log columns.
final class ModelLabTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("unetultra-modellab-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: Model card

    private let fullCard = """
    {
      "contract_version": 1,
      "name": "unet-embed-dinov2s-512",
      "display_name": "UNet + DINOv2-S (512x512)",
      "version": "2026-11-24-a",
      "architecture": "unet_embed",
      "input": {"width": 512, "height": 512, "crop": "center_square"},
      "meta_features": ["log_focus_distance_m", "fx_norm"],
      "outputs": {"mask_logits": {"channels": 1}, "depth": {"units": "m"}},
      "classes": ["background", "airpods_case"],
      "mask_threshold": 0.5,
      "trained_on_sessions": ["20261124-095433-f757"],
      "test_metrics": {"dice": 0.93, "distance_err_pct": null, "size_err_pct": 7.5},
      "size_mb": 45.2,
      "precision": "fp16",
      "notes": "",
      "created_at": "2026-11-24T10:00:00Z",
      "sha256": "abc123",
      "a_key_from_the_future": {"anything": [1, 2, 3]}
    }
    """

    func testFullCardParsesAndIgnoresUnknownKeys() throws {
        let card = try ModelCard.decode(Data(fullCard.utf8))
        try card.validate()
        XCTAssertEqual(card.name, "unet-embed-dinov2s-512")
        XCTAssertEqual(card.contractVersion.major, 1)
        XCTAssertEqual(card.input, ModelCard.Input(width: 512, height: 512, crop: "center_square"))
        XCTAssertEqual(card.metaFeatures, ["log_focus_distance_m", "fx_norm"])
        XCTAssertEqual(card.testMetrics?.dice, 0.93)
        XCTAssertNil(card.testMetrics?.distanceErrPct)
        XCTAssertEqual(card.testMetrics?.sizeErrPct, 7.5)
        XCTAssertEqual(card.sizeMb, 45.2)
        XCTAssertEqual(card.precision, "fp16")
        XCTAssertEqual(card.libraryID, "unet-embed-dinov2s-512-2026-11-24-a")
        XCTAssertTrue(card.hasDepthOutput)
    }

    func testCardDrivesTheDescriptor() throws {
        let card = try ModelCard.decode(Data(fullCard.utf8))
        let descriptor = card.descriptor()
        XCTAssertEqual(descriptor.name, "unet-embed-dinov2s-512")
        XCTAssertEqual(descriptor.displayName, "UNet + DINOv2-S (512x512)")
        XCTAssertEqual(descriptor.inputWidth, 512)
        XCTAssertEqual(descriptor.inputHeight, 512)
        XCTAssertEqual(descriptor.metaFeatures, ["log_focus_distance_m", "fx_norm"])
        XCTAssertEqual(descriptor.maskChannels, 1)
        XCTAssertTrue(descriptor.hasDepth)
        XCTAssertEqual(descriptor.classes, ["background", "airpods_case"])
        XCTAssertEqual(descriptor.crop, "center_square")
        XCTAssertEqual(card.descriptor(displayNameOverride: "Mine").displayName, "Mine")
    }

    func testImageOnlyCardWithoutDepthOrMeta() throws {
        let json = """
        {"contract_version": 1, "name": "focus-only", "input": {"width": 256, "height": 256},
         "meta_features": [], "outputs": {"mask_logits": {"channels": 3}},
         "classes": ["background", "a", "b"]}
        """
        let card = try ModelCard.decode(Data(json.utf8))
        try card.validate()
        let descriptor = card.descriptor()
        XCTAssertTrue(descriptor.metaFeatures.isEmpty)
        XCTAssertFalse(descriptor.hasDepth)
        XCTAssertEqual(descriptor.maskChannels, 3)
        XCTAssertEqual(descriptor.displayName, "focus-only")
        XCTAssertEqual(descriptor.maskThreshold, 0.5)
        XCTAssertEqual(descriptor.crop, CenterSquareCrop.name)
        XCTAssertEqual(card.libraryID, "focus-only")
    }

    func testMinimalCardFallsBackToDefaults() throws {
        let card = try ModelCard.decode(Data(#"{"contract_version": "1.3", "name": "bare", "input": {"width": 512, "height": 512}}"#.utf8))
        try card.validate()
        let descriptor = card.descriptor()
        XCTAssertEqual(descriptor.classes, ModelDescriptor.defaultClasses)
        XCTAssertEqual(descriptor.maskChannels, 1)
        XCTAssertTrue(descriptor.hasDepth)  // no outputs block: read depth if the model has it
        XCTAssertTrue(descriptor.metaFeatures.isEmpty)
    }

    func testUnknownContractVersionIsRefusedWithAReason() throws {
        let json = #"{"contract_version": 2, "name": "future", "input": {"width": 512, "height": 512}}"#
        let card = try ModelCard.decode(Data(json.utf8))
        XCTAssertThrowsError(try card.validate()) { error in
            XCTAssertEqual(error as? ModelCardError, .unsupportedContract(found: "2"))
            let text = error.localizedDescription
            XCTAssertTrue(text.contains("version 2"), text)
            XCTAssertTrue(text.contains("only understands version 1"), text)
        }
    }

    func testBadCardsAreRefused() throws {
        func card(_ extra: String) throws -> ModelCard {
            try ModelCard.decode(Data(#"{"contract_version": 1, "name": "m", "input": {"width": 512, "height": 512}\#(extra)}"#.utf8))
        }
        XCTAssertThrowsError(try card(#", "meta_features": ["phase_of_the_moon"]"#).validate()) { error in
            XCTAssertTrue(error.localizedDescription.contains("phase_of_the_moon"))
        }
        XCTAssertThrowsError(try card(#", "classes": ["background"]"#).validate())
        XCTAssertThrowsError(try card(#", "mask_threshold": 1.5"#).validate())
        let wrongCrop = #"{"contract_version": 1, "name": "m", "input": {"width": 512, "height": 512, "crop": "letterbox"}}"#
        XCTAssertThrowsError(try ModelCard.decode(Data(wrongCrop.utf8)).validate()) { error in
            XCTAssertTrue(error.localizedDescription.contains("letterbox"))
        }
        // A card with no input block cannot be decoded at all, and says what is missing.
        XCTAssertThrowsError(try ModelCard.decode(Data(#"{"contract_version": 1, "name": "m"}"#.utf8))) { error in
            XCTAssertTrue(error.localizedDescription.contains("input"), error.localizedDescription)
        }
    }

    func testLibraryEntryNamesAndConfirmation() {
        var entry = ModelEntry(id: "x", isBundled: false, card: nil, record: ModelRecord(legacyName: "Old"),
                               compiledURL: directory, folder: directory)
        XCTAssertEqual(entry.displayName, "Old")
        XCTAssertFalse(entry.needsConfirmation)
        entry.record.selfTest = .failure("no files")
        XCTAssertTrue(entry.needsConfirmation)
        entry.record.confirmedDespiteFailure = true
        XCTAssertFalse(entry.needsConfirmation)
        entry.record.displayNameOverride = "Mine"
        XCTAssertEqual(entry.displayName, "Mine")
        XCTAssertEqual(ModelStoreLayout.safeID("my model/v1: test"), "my_model_v1__test")
    }

    // MARK: Zip reader

    private func writeZip(_ name: String, entries: [(String, Data)]) throws -> URL {
        let url = directory.appendingPathComponent(name)
        let writer = try ZipWriter(url: url)
        for (entryName, data) in entries {
            try writer.addEntry(name: entryName, data: data)
        }
        try writer.finish()
        return url
    }

    func testZipReaderRoundTripsNestedEntries() throws {
        // Larger than one read chunk, so the streaming path and running checksum are exercised.
        var weights = Data(count: 2_500_000)
        for index in stride(from: 0, to: weights.count, by: 997) { weights[index] = UInt8(index % 251) }
        let entries: [(String, Data)] = [
            ("model_card.json", Data(#"{"name": "m"}"#.utf8)),
            ("model.mlpackage/Manifest.json", Data("{}".utf8)),
            ("model.mlpackage/Data/com.apple.CoreML/weights/weight.bin", weights),
            ("self_test/input.jpg", Data([0xFF, 0xD8, 0xFF, 0xD9])),
            ("self_test/empty.bin", Data()),
        ]
        let zip = try writeZip("bundle.unetmodel", entries: entries)
        let reader = try ZipReader(url: zip)
        XCTAssertEqual(reader.entries.map(\.name), entries.map(\.0))
        XCTAssertEqual(reader.entries[2].size, UInt64(weights.count))

        let out = directory.appendingPathComponent("out", isDirectory: true)
        var lastProgress: (UInt64, UInt64) = (0, 0)
        try reader.extractAll(to: out) { written, total in lastProgress = (written, total) }
        for (name, data) in entries {
            XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent(name)), data, name)
        }
        XCTAssertEqual(lastProgress.0, lastProgress.1)
        XCTAssertEqual(lastProgress.1, UInt64(entries.reduce(0) { $0 + $1.1.count }))
    }

    func testZipReaderDetectsDamage() throws {
        let payload = Data(repeating: 7, count: 5000)
        let zip = try writeZip("damaged.zip", entries: [("a/b.bin", payload)])
        var bytes = try Data(contentsOf: zip)
        bytes[30 + "a/b.bin".utf8.count + 100] ^= 0xFF  // flip one byte of the stored data
        try bytes.write(to: zip)
        let reader = try ZipReader(url: zip)
        XCTAssertThrowsError(try reader.extractAll(to: directory.appendingPathComponent("damaged-out"))) { error in
            XCTAssertEqual(error as? ZipReadError, .checksumMismatch("a/b.bin"))
        }
    }

    func testZipReaderRefusesUnsafePathsCompressionAndNonZips() throws {
        let traversal = try writeZip("traversal.zip", entries: [("ok.txt", Data("ok".utf8)), ("../evil.txt", Data("x".utf8))])
        XCTAssertThrowsError(try ZipReader(url: traversal)) { error in
            XCTAssertEqual(error as? ZipReadError, .unsafePath("../evil.txt"))
        }
        XCTAssertThrowsError(try ZipReader.checkSafe("/etc/passwd"))
        XCTAssertThrowsError(try ZipReader.checkSafe("a/../../b"))
        XCTAssertNoThrow(try ZipReader.checkSafe("model.mlpackage/Data/a..b/weight.bin"))

        // Mark the entry as deflated in the central directory: it must be refused, not mis-read.
        let stored = try writeZip("deflated.zip", entries: [("data.bin", Data(repeating: 1, count: 64))])
        var bytes = try Data(contentsOf: stored)
        let signature = Data([0x50, 0x4B, 0x01, 0x02])
        let position = try XCTUnwrap(bytes.range(of: signature)).lowerBound
        bytes[position + 10] = 8
        try bytes.write(to: stored)
        XCTAssertThrowsError(try ZipReader(url: stored)) { error in
            XCTAssertEqual(error as? ZipReadError, .compressedEntry("data.bin"))
            XCTAssertTrue(error.localizedDescription.contains("without compression"))
        }

        let notZip = directory.appendingPathComponent("plain.txt")
        try Data(repeating: 65, count: 500).write(to: notZip)
        XCTAssertThrowsError(try ZipReader(url: notZip)) { error in
            XCTAssertEqual(error as? ZipReadError, .notAZip)
        }
    }

    func testIncrementalChecksumMatchesTheWholeOne() {
        let data = Data((0..<1000).map { UInt8($0 % 256) })
        var running = CRC32.start
        running = CRC32.update(running, with: data.prefix(333))
        running = CRC32.update(running, with: data.dropFirst(333))
        XCTAssertEqual(CRC32.finish(running), CRC32.checksum(data))
        XCTAssertEqual(CRC32.checksum(Data("123456789".utf8)), 0xCBF4_3926)
    }

    // MARK: .npy

    private func npy(descr: String = "<f4", fortran: Bool = false, shape: String, values: [Float]) -> Data {
        var header = "{'descr': '\(descr)', 'fortran_order': \(fortran ? "True" : "False"), 'shape': \(shape), }"
        // NumPy pads the header with spaces so the data starts on a 64-byte boundary, and ends it with a newline.
        while (10 + header.utf8.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        data.appendLE(UInt16(header.utf8.count))
        data.append(Data(header.utf8))
        for value in values { data.appendLE(value.bitPattern) }
        return data
    }

    func testNpyReaderReadsFloat32Arrays() throws {
        let values: [Float] = [0.25, 0.5, 1.5, -2, 3.25, 100]
        let array = try NpyReader.floats(from: npy(shape: "(2, 3)", values: values))
        XCTAssertEqual(array.shape, [2, 3])
        XCTAssertEqual(array.values, values)
        let vector = try NpyReader.floats(from: npy(shape: "(3,)", values: [1, 2, 3]))
        XCTAssertEqual(vector.shape, [3])
        XCTAssertEqual(vector.values, [1, 2, 3])
    }

    func testNpyReaderRejectsOtherFormats() {
        XCTAssertThrowsError(try NpyReader.floats(from: npy(descr: "<f8", shape: "(1,)", values: [1]))) { error in
            XCTAssertEqual(error as? NpyError, .unsupported("of type <f8"))
        }
        XCTAssertThrowsError(try NpyReader.floats(from: npy(fortran: true, shape: "(1, 1)", values: [1])))
        XCTAssertThrowsError(try NpyReader.floats(from: npy(shape: "(4, 4)", values: [1, 2]))) { error in
            XCTAssertEqual(error as? NpyError, .truncated)
        }
        XCTAssertThrowsError(try NpyReader.floats(from: Data("not an array at all".utf8))) { error in
            XCTAssertEqual(error as? NpyError, .notNpy)
        }
    }

    // MARK: Self-test arithmetic

    /// A 10×10 mask with a 5×4 object, and a depth map with the object at 0.30 m on a 1.00 m background.
    private func scene() -> (mask: [UInt8], depth: [Float]) {
        var mask = [UInt8](repeating: 0, count: 100)
        var depth = [Float](repeating: 1.0, count: 100)
        for row in 2..<7 {
            for column in 3..<7 {
                mask[row * 10 + column] = 1
                depth[row * 10 + column] = 0.30
            }
        }
        return (mask, depth)
    }

    func testSelfTestPassesWhenOutputsMatch() {
        let expected = scene()
        var depth = expected.depth
        for index in depth.indices { depth[index] += 0.002 }  // 0.2 cm off everywhere
        let result = SelfTestComparison.compare(predictedMask: expected.mask, expectedMask: expected.mask,
                                                predictedDepth: depth, expectedDepth: expected.depth)
        XCTAssertTrue(result.passed)
        XCTAssertEqual(result.maskIoU, 1)
        XCTAssertEqual(result.maskDifferentPercent, 0)
        XCTAssertEqual(result.depthMedianDiffCm ?? -1, 0.2, accuracy: 0.01)
        XCTAssertTrue(result.summary.contains("passed"))

        // No depth to compare: the mask alone decides.
        let maskOnly = SelfTestComparison.compare(predictedMask: expected.mask, expectedMask: expected.mask,
                                                  predictedDepth: nil, expectedDepth: nil)
        XCTAssertTrue(maskOnly.passed)
        XCTAssertNil(maskOnly.depthMedianDiffCm)

        // Nothing expected and nothing found is a pass too.
        let empty = [UInt8](repeating: 0, count: 100)
        XCTAssertTrue(SelfTestComparison.compare(predictedMask: empty, expectedMask: empty,
                                                 predictedDepth: nil, expectedDepth: nil).passed)
    }

    func testSelfTestFailsOnAMaskMismatch() {
        let expected = scene()
        var predicted = expected.mask
        for row in 2..<7 { predicted[row * 10 + 6] = 0 }  // one of four columns missing: IoU 15/20
        let result = SelfTestComparison.compare(predictedMask: predicted, expectedMask: expected.mask,
                                                predictedDepth: expected.depth, expectedDepth: expected.depth)
        XCTAssertFalse(result.passed)
        XCTAssertEqual(result.maskIoU, 0.75, accuracy: 1e-9)
        XCTAssertEqual(result.maskDifferentPercent, 5, accuracy: 1e-9)
        XCTAssertTrue(result.summary.contains("FAILED"))
        XCTAssertTrue(result.summary.contains("pre-processing"))
    }

    func testSelfTestFailsOnADepthMismatch() {
        let expected = scene()
        var depth = expected.depth
        for index in depth.indices where expected.mask[index] != 0 { depth[index] += 0.05 }  // object 5 cm off
        let result = SelfTestComparison.compare(predictedMask: expected.mask, expectedMask: expected.mask,
                                                predictedDepth: depth, expectedDepth: expected.depth)
        XCTAssertFalse(result.passed)
        XCTAssertEqual(result.maskIoU, 1)
        XCTAssertEqual(result.depthMedianDiffCm ?? -1, 5, accuracy: 0.01)
        XCTAssertEqual(result.depthMaxDiffCm ?? -1, 5, accuracy: 0.01)

        // A background-only error does not fail the test: only the object's depth is compared.
        var background = expected.depth
        for index in background.indices where expected.mask[index] == 0 { background[index] += 0.5 }
        XCTAssertTrue(SelfTestComparison.compare(predictedMask: expected.mask, expectedMask: expected.mask,
                                                 predictedDepth: background, expectedDepth: expected.depth).passed)
    }

    func testSelfTestReportsSizeProblems() {
        let expected = scene()
        let result = SelfTestComparison.compare(predictedMask: [0, 1], expectedMask: expected.mask,
                                                predictedDepth: nil, expectedDepth: nil)
        XCTAssertFalse(result.passed)
        XCTAssertNotNil(result.problem)
        XCTAssertTrue(result.summary.contains("could not run"))
    }

    func testSelfTestResultSurvivesStorage() throws {
        let result = SelfTestResult(passed: true, maskDifferentPercent: 0.1, maskIoU: 0.99,
                                    depthMedianDiffCm: 0.2, depthMaxDiffCm: 1.4)
        let record = ModelRecord(importedAt: Date(timeIntervalSince1970: 1_800_000_000), selfTest: result,
                                 displayNameOverride: "Mine")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        XCTAssertEqual(try decoder.decode(ModelRecord.self, from: encoder.encode(record)), record)
    }

    // MARK: Latency and compute units

    func testLatencyKeepsTheLastThirtyRuns() {
        var stats = LatencyStats()
        XCTAssertNil(stats.median)
        XCTAssertEqual(stats.summary, "no runs yet")
        for value in 1...40 { stats.add(Double(value)) }
        XCTAssertEqual(stats.count, 30)  // 11...40
        XCTAssertEqual(stats.last, 40)
        XCTAssertEqual(stats.median ?? -1, 25.5, accuracy: 1e-9)
        XCTAssertEqual(stats.p95 ?? -1, 38.55, accuracy: 1e-9)
        stats.add(.nan)
        XCTAssertEqual(stats.count, 30)

        var single = LatencyStats()
        single.add(12)
        XCTAssertEqual(single.median, 12)
        XCTAssertEqual(single.p95, 12)
    }

    func testComputeUnitsNamesAreStable() {
        XCTAssertEqual(ComputeUnitsOption.allCases.map(\.rawValue),
                       ["all", "cpu_and_neural_engine", "cpu_and_gpu", "cpu_only"])
        XCTAssertEqual(ComputeUnitsOption(rawValue: "cpu_only"), .cpuOnly)
    }

    // MARK: Log columns

    func testLogRowsRecordModelVersionArchitectureAndComputeUnits() throws {
        XCTAssertEqual(Array(InferenceLogRow.header.suffix(3)), ["model_version", "architecture", "compute_units"])
        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: 512, outputHeight: 512)
        var result = InferenceResult(
            modelName: "unet-multi-512", modelDisplayName: "UNet", found: true, classId: 1, classLabel: "airpods_case",
            pixels: [1, 2, 3], maskWidth: 512, maskHeight: 512, containsTap: true,
            measurement: ObjectMeasurement(widthM: 0.054, heightM: 0.046, distanceM: 0.312), depthSource: .model,
            lidarDistanceM: 0.308, focusDistanceM: 0.31, lensPosition: 0.4123, latencyMs: 11.6, crop: crop)
        result.modelVersion = "2026-11-24-a"
        result.architecture = "unet_multi"
        result.computeUnits = ComputeUnitsOption.cpuAndNeuralEngine.rawValue
        let row = InferenceLogRow(timestamp: "2026-01-01T12:00:00Z", deviceModel: "iPhone16,1", result: result,
                                  measuredWidthCm: nil, measuredHeightCm: nil, measuredDistanceCm: nil, notes: "")
        XCTAssertEqual(row.fields.count, InferenceLogRow.header.count)
        XCTAssertEqual(Array(row.fields.suffix(3)), ["2026-11-24-a", "unet_multi", "cpu_and_neural_engine"])
        XCTAssertEqual(row.fields[1], "unet-multi-512")
    }

    func testAnOldLogIsLeftAloneAndANewFileIsStarted() throws {
        let header = InferenceLogRow.header
        let fresh = ResultLog.resolve(directory: directory, baseName: "inference_log", header: header)
        XCTAssertEqual(fresh.fileURL.lastPathComponent, "inference_log.csv")
        try fresh.append(header.map { _ in "x" })
        XCTAssertEqual(ResultLog.resolve(directory: directory, baseName: "inference_log", header: header).fileURL,
                       fresh.fileURL)

        // An existing file with the older, shorter header must not get rows with more columns.
        let old = directory.appendingPathComponent("old", isDirectory: true)
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        let oldHeader = Array(header.dropLast(3))
        try Data((CSV.line(oldHeader) + "\n1,2,3\n").utf8).write(to: old.appendingPathComponent("inference_log.csv"))
        let next = ResultLog.resolve(directory: old, baseName: "inference_log", header: header)
        XCTAssertEqual(next.fileURL.lastPathComponent, "inference_log_v2.csv")
        try next.append(header.map { _ in "y" })
        XCTAssertEqual(ResultLog.resolve(directory: old, baseName: "inference_log", header: header).fileURL, next.fileURL)
        XCTAssertEqual(next.rowCount(), 1)
    }
}
