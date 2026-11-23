import XCTest
@testable import UNetUltra

/// Tests for the inference pipeline's pure logic. No camera and no model file are needed.
final class InferenceTests: XCTestCase {
    private let imageIntrinsics = Intrinsics(fx: 1450, fy: 1450, cx: 972, cy: 708)

    // MARK: Centre-square crop

    func testCropMapsPointsBetweenImageAndModel() {
        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: 512, outputHeight: 512)
        XCTAssertEqual(crop.side, 1440)
        XCTAssertEqual(crop.x0, 240)
        XCTAssertEqual(crop.y0, 0)

        let centre = crop.toOutput(x: 959.5, y: 719.5)
        XCTAssertEqual(centre.x, 255.5, accuracy: 1e-9)
        XCTAssertEqual(centre.y, 255.5, accuracy: 1e-9)
        XCTAssertEqual(crop.toOutput(x: 239.5, y: 0).x, -0.5, accuracy: 1e-9)
        XCTAssertEqual(crop.toOutput(x: 1679.5, y: 0).x, 511.5, accuracy: 1e-9)

        // The inverse takes model pixels back to image pixels.
        for point in [(0.0, 0.0), (255.5, 255.5), (511.0, 37.25)] {
            let source = crop.toSource(x: point.0, y: point.1)
            let back = crop.toOutput(x: source.x, y: source.y)
            XCTAssertEqual(back.x, point.0, accuracy: 1e-9)
            XCTAssertEqual(back.y, point.1, accuracy: 1e-9)
        }
        XCTAssertEqual(crop.toSource(x: 255.5, y: 255.5).x, 959.5, accuracy: 1e-9)
    }

    func testCropScalesIntrinsics() {
        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: 512, outputHeight: 512)
        let scaled = crop.intrinsics(imageIntrinsics)
        let s = 512.0 / 1440.0
        XCTAssertEqual(scaled.fx, 1450 * s, accuracy: 1e-9)
        XCTAssertEqual(scaled.fy, 1450 * s, accuracy: 1e-9)
        XCTAssertEqual(scaled.cx, (972 - 240 + 0.5) * s - 0.5, accuracy: 1e-9)
        XCTAssertEqual(scaled.cy, (708 + 0.5) * s - 0.5, accuracy: 1e-9)
    }

    func testCropKnowsWhichTapsAreInside() {
        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: 512, outputHeight: 512)
        XCTAssertTrue(crop.contains(x: 239.5, y: 700))
        XCTAssertFalse(crop.contains(x: 239.4, y: 700))
        XCTAssertTrue(crop.contains(x: 1679.4, y: 700))
        XCTAssertFalse(crop.contains(x: 1679.5, y: 700))
        XCTAssertTrue(crop.contains(x: 960, y: 0))
        XCTAssertFalse(crop.contains(x: 960, y: 1439.5))

        XCTAssertTrue(CenterSquareCrop.containsDevicePoint(x: 0.5, y: 0.5, imageWidth: 1920, imageHeight: 1440))
        XCTAssertFalse(CenterSquareCrop.containsDevicePoint(x: 0.1, y: 0.5, imageWidth: 1920, imageHeight: 1440))
        XCTAssertFalse(CenterSquareCrop.containsDevicePoint(x: 0.9, y: 0.5, imageWidth: 1920, imageHeight: 1440))

        let rect = crop.normalizedRect
        XCTAssertEqual(rect.minX, 0.125, accuracy: 1e-12)
        XCTAssertEqual(rect.maxX, 0.875, accuracy: 1e-12)
        XCTAssertEqual(rect.minY, 0, accuracy: 1e-12)
        XCTAssertEqual(rect.maxY, 1, accuracy: 1e-12)
    }

    func testDepthMapIsCroppedToItsOwnSquare() {
        let crop = CenterSquareCrop(sourceWidth: 256, sourceHeight: 192, outputWidth: 512, outputHeight: 512)
        XCTAssertEqual(crop.x0, 32)
        XCTAssertEqual(crop.side, 192)
        XCTAssertNoThrow(try CenterSquareCrop.checkSameAspect(imageWidth: 1920, imageHeight: 1440,
                                                              depthWidth: 256, depthHeight: 192))
        XCTAssertThrowsError(try CenterSquareCrop.checkSameAspect(imageWidth: 1920, imageHeight: 1440,
                                                                  depthWidth: 256, depthHeight: 144)) { error in
            XCTAssertEqual(error as? CropError,
                           .aspectMismatch(imageWidth: 1920, imageHeight: 1440, depthWidth: 256, depthHeight: 144))
        }
        XCTAssertThrowsError(try CenterSquareCrop.validate(name: "letterbox")) { error in
            XCTAssertEqual(error as? CropError, .unsupportedCrop("letterbox"))
        }
        XCTAssertNoThrow(try CenterSquareCrop.validate(name: "center_square"))
    }

    // MARK: Lens calibration

    func testCalibrationInterpolatesInInverseDistanceAndClamps() throws {
        let calibration = LensCalibration(lensPosition: [0.2, 0.5, 0.8], distanceM: [0.2, 0.4, 0.8])
        XCTAssertNil(calibration.problem)
        let middle = try XCTUnwrap(calibration.focusDistance(lensPosition: 0.35))
        XCTAssertEqual(middle, 1 / ((1 / 0.2 + 1 / 0.4) / 2), accuracy: 1e-9)  // 0.2667
        XCTAssertEqual(try XCTUnwrap(calibration.focusDistance(lensPosition: 0.5)), 0.4, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(calibration.focusDistance(lensPosition: 0.0)), 0.2, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(calibration.focusDistance(lensPosition: 1.0)), 0.8, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(calibration.focusDistance(lensPosition: 0.65)),
                       1 / ((1 / 0.4 + 1 / 0.8) / 2), accuracy: 1e-9)
    }

    func testCalibrationDecodesPythonJSON() throws {
        let python = """
        {
          "format_version": 1,
          "kind": "lens_position_to_distance",
          "interpolation": "linear_in_inverse_distance",
          "device_model": "iPhone15,3",
          "lens_position": [0.17, 0.58, 0.82],
          "distance_m": [0.15, 0.3, 0.7],
          "n_samples": 42,
          "rmse_m": 0.0029,
          "created_at": "2026-11-23T16:00:00+00:00"
        }
        """
        let calibration = try JSONDecoder().decode(LensCalibration.self, from: Data(python.utf8))
        XCTAssertNil(calibration.problem)
        XCTAssertEqual(calibration.deviceModel, "iPhone15,3")
        XCTAssertEqual(calibration.nSamples, 42)
        XCTAssertEqual(try XCTUnwrap(calibration.focusDistance(lensPosition: 0.58)), 0.3, accuracy: 1e-9)

        let broken = LensCalibration(lensPosition: [0.2, 0.5], distanceM: [0.2])
        XCTAssertNotNil(broken.problem)
        XCTAssertNil(broken.focusDistance(lensPosition: 0.3))
    }

    // MARK: Meta features

    func testMetaFeaturesFollowTheCardOrder() throws {
        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: 512, outputHeight: 512)
        let fxModel = crop.intrinsics(imageIntrinsics).fx  // 1450 · 512 / 1440
        let inputs = MetaFeatureInputs(focusDistanceM: 0.3, fxModel: fxModel, lensPosition: 0.42)
        let values = try MetaFeatureProvider.values(for: ["fx_norm", "log_focus_distance_m", "lens_position_raw"],
                                                    inputs: inputs)
        XCTAssertEqual(values.count, 3)
        XCTAssertEqual(values[0], Float(1450.0 * 512 / 1440 / 1000 - 1), accuracy: 1e-6)
        XCTAssertEqual(values[1], Float(log(0.3)), accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.42, accuracy: 1e-6)
        XCTAssertEqual(try MetaFeatureProvider.values(for: [], inputs: inputs), [])
    }

    func testMetaFeatureErrors() {
        let inputs = MetaFeatureInputs(focusDistanceM: nil, fxModel: 500, lensPosition: 0.4)
        XCTAssertThrowsError(try MetaFeatureProvider.values(for: ["fx_norm", "zoom"], inputs: inputs)) { error in
            XCTAssertEqual(error as? MetaFeatureError, .unknownFeature("zoom"))
            XCTAssertTrue(error.localizedDescription.contains("zoom"))
        }
        XCTAssertThrowsError(try MetaFeatureProvider.values(for: ["log_focus_distance_m"], inputs: inputs)) { error in
            XCTAssertEqual(error as? MetaFeatureError, .missingCalibration("log_focus_distance_m"))
        }
    }

    // MARK: Connected components

    /// A class map with a filled rectangle of one class.
    private func fill(_ map: inout [UInt8], width: Int, columns: ClosedRange<Int>, rows: ClosedRange<Int>, classId: UInt8 = 1) {
        for row in rows {
            for column in columns { map[row * width + column] = classId }
        }
    }

    func testConnectedComponentsAndTapSelection() {
        let width = 64, height = 48
        var map = [UInt8](repeating: 0, count: width * height)
        fill(&map, width: width, columns: 5...24, rows: 5...14)     // 200 px
        fill(&map, width: width, columns: 40...49, rows: 30...35)   // 60 px
        fill(&map, width: width, columns: 60...61, rows: 2...3)     // 4 px speck
        map[15 * width + 25] = 1                                     // diagonal neighbour only: not joined

        let components = ConnectedComponents.components(classMap: map, width: width, height: height)
        XCTAssertEqual(components.map(\.area), [200, 60])

        let tapOnSmall = 32 * width + 45
        let hit = ConnectedComponents.select(components, tapIndex: tapOnSmall)
        XCTAssertEqual(hit?.component.area, 60)
        XCTAssertEqual(hit?.containsTap, true)

        let miss = ConnectedComponents.select(components, tapIndex: 0)
        XCTAssertEqual(miss?.component.area, 200)
        XCTAssertEqual(miss?.containsTap, false)
        XCTAssertEqual(ConnectedComponents.select(components, tapIndex: nil)?.component.area, 200)
        XCTAssertNil(ConnectedComponents.select([], tapIndex: 3))

        // Touching blobs of different classes stay separate.
        var twoClasses = [UInt8](repeating: 0, count: width * height)
        fill(&twoClasses, width: width, columns: 0...9, rows: 0...9, classId: 1)
        fill(&twoClasses, width: width, columns: 10...19, rows: 0...9, classId: 2)
        let split = ConnectedComponents.components(classMap: twoClasses, width: width, height: height)
        XCTAssertEqual(Set(split.map(\.classId)), [1, 2])
        XCTAssertEqual(split.map(\.area), [100, 100])
    }

    // MARK: Geometry

    /// A 51 × 53 pixel rectangle at 0.4 m in a 512 × 384 map, as in the Python tests.
    private func rectangleScene() -> (pixels: [Int], depth: [Float], width: Int, intrinsics: Intrinsics) {
        let width = 512, height = 384
        var depth = [Float](repeating: 1.2, count: width * height)
        var pixels: [Int] = []
        for row in 124...176 {
            for column in 275...325 {
                pixels.append(row * width + column)
                depth[row * width + column] = 0.4
            }
        }
        return (pixels, depth, width, Intrinsics(fx: 400, fy: 410, cx: 250, cy: 190))
    }

    func testObjectGeometryMeasuresTheFullPixelFootprint() throws {
        let scene = rectangleScene()
        let measured = try XCTUnwrap(ObjectGeometry.measure(
            pixels: scene.pixels, mapWidth: scene.width, depth: scene.depth, fallbackDistanceM: nil,
            intrinsics: scene.intrinsics))
        XCTAssertEqual(measured.distanceM, 0.4, accuracy: 1e-6)
        XCTAssertEqual(measured.widthM, 51 * 0.4 / 400, accuracy: 1e-6)
        XCTAssertEqual(measured.heightM, 53 * 0.4 / 410, accuracy: 1e-6)
    }

    func testObjectGeometryIgnoresBackgroundDepthOnTheEdge() throws {
        var scene = rectangleScene()
        for row in 124...176 {
            for column in 323...325 { scene.depth[row * scene.width + column] = 1.2 }
        }
        let measured = try XCTUnwrap(ObjectGeometry.measure(
            pixels: scene.pixels, mapWidth: scene.width, depth: scene.depth, fallbackDistanceM: nil,
            intrinsics: scene.intrinsics))
        XCTAssertEqual(measured.distanceM, 0.4, accuracy: 1e-6)
        XCTAssertEqual(measured.widthM, 51 * 0.4 / 400, accuracy: 1e-6)
        XCTAssertEqual(measured.heightM, 53 * 0.4 / 410, accuracy: 1e-6)
    }

    func testObjectGeometryFallsBackToFocusDistanceWithoutDepth() throws {
        let scene = rectangleScene()
        let measured = try XCTUnwrap(ObjectGeometry.measure(
            pixels: scene.pixels, mapWidth: scene.width, depth: nil, fallbackDistanceM: 0.3,
            intrinsics: scene.intrinsics))
        XCTAssertEqual(measured.distanceM, 0.3, accuracy: 1e-9)
        XCTAssertEqual(measured.widthM, 51 * 0.3 / 400, accuracy: 1e-9)
        XCTAssertNil(ObjectGeometry.measure(pixels: scene.pixels, mapWidth: scene.width, depth: nil,
                                            fallbackDistanceM: nil, intrinsics: scene.intrinsics))
        XCTAssertNil(ObjectGeometry.measure(pixels: [], mapWidth: scene.width, depth: scene.depth,
                                            fallbackDistanceM: nil, intrinsics: scene.intrinsics))
    }

    func testMediansAndQuantiles() {
        XCTAssertEqual(ObjectGeometry.lowerMedian([4, 1, 3, 2]), 2)
        XCTAssertEqual(ObjectGeometry.median([4, 1, 3, 2]), 2.5)
        XCTAssertEqual(ObjectGeometry.lowerMedian([5, 1, 3]), 3)
        XCTAssertNil(ObjectGeometry.median([]))
        XCTAssertEqual(ObjectGeometry.quantile(sorted: [0, 10, 20, 30, 40], 0.5), 20, accuracy: 1e-12)
        XCTAssertEqual(ObjectGeometry.quantile(sorted: [0, 10, 20, 30, 40], 0.1), 4, accuracy: 1e-12)
        XCTAssertEqual(ObjectGeometry.quantile(sorted: [0, 10], 1), 10, accuracy: 1e-12)
    }

    // MARK: Whole analysis

    private func descriptor(hasDepth: Bool) -> ModelDescriptor {
        ModelDescriptor(name: "test", displayName: "Test model", inputWidth: 512, inputHeight: 512,
                        metaFeatures: ModelDescriptor.defaultMetaFeatures, maskChannels: 1, hasDepth: hasDepth,
                        classes: ["background", "airpods_case"])
    }

    /// A 1920 × 1440 frame whose LiDAR map reads 0.5 m everywhere.
    private func frameGeometry(lidarHeight: Int = 192) -> FrameGeometry {
        FrameGeometry(imageWidth: 1920, imageHeight: 1440, intrinsicsImage: imageIntrinsics, lensPosition: 0.4,
                      lidar: [Float](repeating: 0.5, count: 256 * lidarHeight), lidarWidth: 256, lidarHeight: lidarHeight)
    }

    func testAnalysisMeasuresTheTappedObjectWithModelDepth() throws {
        let size = 512
        var map = [UInt8](repeating: 0, count: size * size)
        fill(&map, width: size, columns: 100...159, rows: 200...249)   // 60 × 50, the larger blob
        fill(&map, width: size, columns: 300...339, rows: 100...129)   // 40 × 30
        var depth = [Float](repeating: 1.0, count: size * size)
        for row in 100...129 {
            for column in 300...339 { depth[row * size + column] = 0.25 }
        }
        let output = ModelOutput(classMap: map, depth: depth, width: size, height: size)
        let geometry = frameGeometry()
        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: size, outputHeight: size)

        // Tap the centre of the smaller blob, given in full-image pixels.
        let tap = crop.toSource(x: 320, y: 115)
        let result = try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor(hasDepth: true), geometry: geometry, focusDistanceM: 0.3,
            tapImagePoint: tap, latencyMs: 12)
        XCTAssertTrue(result.found)
        XCTAssertTrue(result.containsTap)
        XCTAssertEqual(result.areaPx, 40 * 30)
        XCTAssertEqual(result.classLabel, "airpods_case")
        XCTAssertEqual(result.depthSource, .model)
        let fx = 1450.0 * 512 / 1440
        let measurement = try XCTUnwrap(result.measurement)
        XCTAssertEqual(measurement.distanceM, 0.25, accuracy: 1e-6)
        XCTAssertEqual(measurement.widthM, 40 * 0.25 / fx, accuracy: 1e-6)
        XCTAssertEqual(measurement.heightM, 30 * 0.25 / fx, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(result.lidarDistanceM), 0.5, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(result.distanceDifferenceM), -0.25, accuracy: 1e-6)
        XCTAssertEqual(result.latencyMs, 12)

        // The label position is the blob's centre in normalized full-frame coordinates.
        let centroid = try XCTUnwrap(result.centroid)
        let expected = crop.toSource(x: 319.5, y: 114.5)
        XCTAssertEqual(centroid.x, (expected.x + 0.5) / 1920, accuracy: 1e-9)
        XCTAssertEqual(centroid.y, (expected.y + 0.5) / 1440, accuracy: 1e-9)

        // A tap outside the square is not a hit: the largest blob is measured instead.
        let outside = try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor(hasDepth: true), geometry: geometry, focusDistanceM: 0.3,
            tapImagePoint: (x: 100, y: 700), latencyMs: 1)
        XCTAssertFalse(outside.containsTap)
        XCTAssertEqual(outside.areaPx, 60 * 50)
    }

    func testAnalysisUsesFocusDistanceWhenTheModelHasNoDepth() throws {
        let size = 512
        var map = [UInt8](repeating: 0, count: size * size)
        fill(&map, width: size, columns: 200...259, rows: 220...269)
        let output = ModelOutput(classMap: map, depth: nil, width: size, height: size)
        let result = try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor(hasDepth: false), geometry: frameGeometry(), focusDistanceM: 0.3,
            tapImagePoint: nil, latencyMs: 5)
        XCTAssertEqual(result.depthSource, .focusDistance)
        let measurement = try XCTUnwrap(result.measurement)
        XCTAssertEqual(measurement.distanceM, 0.3, accuracy: 1e-9)
        XCTAssertEqual(measurement.widthM, 60 * 0.3 / (1450.0 * 512 / 1440), accuracy: 1e-9)

        // No depth output and no calibration: the mask is still reported, without numbers.
        let blind = try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor(hasDepth: false), geometry: frameGeometry(), focusDistanceM: nil,
            tapImagePoint: nil, latencyMs: 5)
        XCTAssertTrue(blind.found)
        XCTAssertNil(blind.measurement)
        XCTAssertEqual(blind.depthSource, DepthSource.none)
    }

    func testAnalysisReportsNothingFoundAndRejectsBadSetups() throws {
        let size = 512
        var map = [UInt8](repeating: 0, count: size * size)
        fill(&map, width: size, columns: 10...13, rows: 10...13)  // 16 px: below the minimum
        let output = ModelOutput(classMap: map, depth: nil, width: size, height: size)
        let empty = try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor(hasDepth: false), geometry: frameGeometry(), focusDistanceM: 0.3,
            tapImagePoint: nil, latencyMs: 5)
        XCTAssertFalse(empty.found)
        XCTAssertNil(empty.measurement)
        XCTAssertNil(empty.centroid)

        var letterbox = descriptor(hasDepth: false)
        letterbox.crop = "letterbox"
        XCTAssertThrowsError(try InferenceAnalysis.analyse(
            output: output, descriptor: letterbox, geometry: frameGeometry(), focusDistanceM: 0.3,
            tapImagePoint: nil, latencyMs: 5)) { error in
            XCTAssertEqual(error as? CropError, .unsupportedCrop("letterbox"))
        }
        XCTAssertThrowsError(try InferenceAnalysis.analyse(
            output: output, descriptor: descriptor(hasDepth: false), geometry: frameGeometry(lidarHeight: 144),
            focusDistanceM: 0.3, tapImagePoint: nil, latencyMs: 5))
    }

    func testLidarReferenceSkipsMissingReadings() throws {
        // 256 × 192 depth map: the left half of the centre square has no reading.
        var lidar = [Float](repeating: 0.6, count: 256 * 192)
        for row in 0..<192 {
            for column in 0..<128 { lidar[row * 256 + column] = row % 2 == 0 ? 0 : .nan }
        }
        let size = 512
        let leftPixels = (200..<220).flatMap { row in (10..<40).map { row * size + $0 } }
        XCTAssertNil(ObjectGeometry.lidarDistance(pixels: leftPixels, mapWidth: size, mapHeight: size,
                                                  lidar: lidar, lidarWidth: 256, lidarHeight: 192))
        let rightPixels = (200..<220).flatMap { row in (400..<440).map { row * size + $0 } }
        let reading = try XCTUnwrap(ObjectGeometry.lidarDistance(
            pixels: rightPixels, mapWidth: size, mapHeight: size, lidar: lidar, lidarWidth: 256, lidarHeight: 192))
        XCTAssertEqual(reading, 0.6, accuracy: 1e-6)
    }

    // MARK: Model description and output decoding

    func testDescriptorHelpers() {
        XCTAssertEqual(ModelDescriptor.parseClasses(json: "[\"background\", \"airpods_case\"]") ?? [],
                       ["background", "airpods_case"])
        XCTAssertNil(ModelDescriptor.parseClasses(json: "not json"))
        XCTAssertNil(ModelDescriptor.parseClasses(json: nil))
        let d = descriptor(hasDepth: true)
        XCTAssertEqual(d.crop, "center_square")
        XCTAssertEqual(d.label(forClassId: 1), "airpods_case")
        XCTAssertEqual(d.label(forClassId: 9), "object")
        XCTAssertEqual(d.logitThreshold, 0, accuracy: 1e-6)  // threshold 0.5
    }

    func testClassMapFromLogits() {
        // One channel: object where the logit beats logit(threshold).
        XCTAssertEqual(CoreMLSegmentationModel.classMap(logits: [-2, 0.1, 3, -0.1], channels: 1, count: 4,
                                                        logitThreshold: 0), [0, 1, 1, 0])
        // Three channels, laid out channel after channel: arg-max, channel 0 is background.
        let logits: [Float] = [5, 0, 0, 1,   1, 4, 0, 1,   0, 1, 6, 1]
        XCTAssertEqual(CoreMLSegmentationModel.classMap(logits: logits, channels: 3, count: 4, logitThreshold: 0),
                       [0, 1, 2, 0])
    }

    func testHalfFloatDecoding() {
        XCTAssertEqual(MultiArrayReader.halfToFloat(0x3C00), 1.0)
        XCTAssertEqual(MultiArrayReader.halfToFloat(0xC000), -2.0)
        XCTAssertEqual(MultiArrayReader.halfToFloat(0x3800), 0.5)
        XCTAssertEqual(MultiArrayReader.halfToFloat(0x0000), 0.0)
        XCTAssertTrue(MultiArrayReader.halfToFloat(0x7E00).isNaN)
    }

    // MARK: CSV log

    func testCSVQuoting() {
        XCTAssertEqual(CSV.escape("plain"), "plain")
        XCTAssertEqual(CSV.line(["a", "b,c", "say \"hi\"", "two\nlines", ""]),
                       "a,\"b,c\",\"say \"\"hi\"\"\",\"two\nlines\",")
    }

    func testResultLogAppendsRowsUnderOneHeader() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let log = ResultLog(fileURL: directory.appendingPathComponent("log.csv"), header: InferenceLogRow.header)
        XCTAssertEqual(log.rowCount(), 0)

        let crop = CenterSquareCrop(sourceWidth: 1920, sourceHeight: 1440, outputWidth: 512, outputHeight: 512)
        let result = InferenceResult(
            modelName: "unet, v1", modelDisplayName: "UNet", found: true, classId: 1, classLabel: "airpods_case",
            pixels: [1, 2, 3], maskWidth: 512, maskHeight: 512, containsTap: true,
            measurement: ObjectMeasurement(widthM: 0.054, heightM: 0.046, distanceM: 0.312), depthSource: .model,
            lidarDistanceM: 0.308, focusDistanceM: 0.31, lensPosition: 0.4123, latencyMs: 11.6, crop: crop)
        let row = InferenceLogRow(timestamp: "2026-01-01T12:00:00Z", deviceModel: "iPhone15,3", result: result,
                                  measuredWidthCm: 5.5, measuredHeightCm: nil, measuredDistanceCm: 31,
                                  notes: "first line\nsecond line")
        XCTAssertEqual(row.fields.count, InferenceLogRow.header.count)
        try log.append(row.fields)
        try log.append(row.fields)
        XCTAssertEqual(log.rowCount(), 2)  // the line break inside the quoted note does not add a row

        let text = try String(contentsOf: log.fileURL, encoding: .utf8)
        XCTAssertTrue(text.hasPrefix("timestamp,model_name,device_model,"))
        XCTAssertTrue(text.contains("\"unet, v1\",\"iPhone15,3\",0.4123,31.00,31.20,5.40,4.60,30.80,0.40,3,11.6,airpods_case,5.50,,31.00,"))
    }
}
