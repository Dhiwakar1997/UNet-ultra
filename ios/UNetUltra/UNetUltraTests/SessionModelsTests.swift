import XCTest
@testable import UNetUltra

final class SessionModelsTests: XCTestCase {
    private func sampleMeta(objects: [ObjectAnnotation]) -> SampleMeta {
        SampleMeta(
            sampleId: "0001",
            timestamp: "2026-01-01T12:00:00Z",
            imageSize: [1920, 1440],
            depthSize: [256, 192],
            intrinsicsImage: Intrinsics(fx: 1450, fy: 1450, cx: 972, cy: 708),
            intrinsicsDepth: Intrinsics(fx: 193.3, fy: 193.3, cx: 129.2, cy: 94.0),
            intrinsicsReferenceSize: [4032, 3024],
            lensPosition: 0.42,
            focusSettled: true,
            focusTapIndex: 0,
            deviceOrientation: "portrait",
            exposure: Exposure(iso: 100, durationS: 1.0 / 60.0),
            backgroundOnly: objects.isEmpty,
            objects: objects
        )
    }

    private func jsonObject(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testMetaUsesContractKeysAndRoundTrips() throws {
        let object = ObjectAnnotation(classId: 1, label: "airpods_case", tapPointImage: [960, 720],
                                      lidarDistanceAtTapM: 0.31)
        let meta = sampleMeta(objects: [object])
        let data = try SessionJSON.encoder().encode(meta)

        let json = try jsonObject(data)
        let expectedKeys: Set<String> = [
            "sample_id", "timestamp", "image_size", "depth_size", "intrinsics_image", "intrinsics_depth",
            "intrinsics_reference_size", "lens_position", "focus_settled", "focus_tap_index",
            "device_orientation", "exposure", "background_only", "objects", "notes",
        ]
        XCTAssertEqual(Set(json.keys), expectedKeys)
        XCTAssertEqual(try SessionJSON.decoder().decode(SampleMeta.self, from: data), meta)
    }

    func testUnmeasuredValuesAreWrittenAsNull() throws {
        let object = ObjectAnnotation(classId: 1, label: "airpods_case", tapPointImage: [10, 20])
        let json = try jsonObject(try SessionJSON.encoder().encode(sampleMeta(objects: [object])))
        let first = try XCTUnwrap((json["objects"] as? [[String: Any]])?.first)

        XCTAssertEqual(Set(first.keys), [
            "class_id", "label", "tap_point_image", "extra_points", "lidar_distance_at_tap_m",
            "true_size_cm", "facing", "true_distance_cm",
        ])
        XCTAssertTrue(first["facing"] is NSNull)
        XCTAssertTrue(first["true_distance_cm"] is NSNull)
        let size = try XCTUnwrap(first["true_size_cm"] as? [String: Any])
        XCTAssertTrue(size["width"] is NSNull && size["height"] is NSNull)
    }

    func testBackgroundOnlySampleHasEmptyObjects() throws {
        let json = try jsonObject(try SessionJSON.encoder().encode(sampleMeta(objects: [])))
        XCTAssertEqual(json["background_only"] as? Bool, true)
        XCTAssertEqual((json["objects"] as? [Any])?.count, 0)
    }

    func testManifestUsesContractKeysAndRoundTrips() throws {
        var manifest = SessionManifest(
            sessionId: "abc123", split: .train,
            classes: [SessionFormat.backgroundClass, "airpods_case"],
            deviceModel: "iPhone15,3", iosVersion: "17.5", appVersion: "0.1.0",
            createdAt: "2026-01-01T12:00:00Z"
        )
        manifest.sampleIds = ["0001"]
        manifest.maskStatus = ["0001": .pending]
        let data = try SessionJSON.encoder().encode(manifest)

        let json = try jsonObject(data)
        XCTAssertEqual(Set(json.keys), [
            "format_version", "session_id", "split", "classes", "device_model", "ios_version",
            "app_version", "created_at", "sample_ids", "mask_status",
        ])
        XCTAssertEqual(json["split"] as? String, "train")
        XCTAssertEqual((json["mask_status"] as? [String: String])?["0001"], "pending")
        XCTAssertEqual(try SessionJSON.decoder().decode(SessionManifest.self, from: data), manifest)
    }

    func testDecodesMetaWrittenByPython() throws {
        let python = """
        {"sample_id": "0000", "timestamp": "2026-01-01T12:00:00+00:00", "image_size": [960, 720],
         "depth_size": [256, 192],
         "intrinsics_image": {"fx": 725.0, "fy": 725.0, "cx": 485.75, "cy": 353.75},
         "intrinsics_depth": {"fx": 193.3, "fy": 193.3, "cx": 129.2, "cy": 94.0},
         "intrinsics_reference_size": [1920, 1440], "lens_position": 0.4, "focus_settled": true,
         "focus_tap_index": 0, "device_orientation": "landscapeRight",
         "exposure": {"iso": 100, "duration_s": 0.0166}, "background_only": false,
         "objects": [{"class_id": 1, "label": "airpods_case", "tap_point_image": [400.5, 300.0],
                      "extra_points": [], "lidar_distance_at_tap_m": 0.3,
                      "true_size_cm": {"width": 5.5, "height": null}, "facing": null,
                      "true_distance_cm": null}],
         "notes": "synthetic"}
        """
        let meta = try SessionJSON.decoder().decode(SampleMeta.self, from: Data(python.utf8))
        XCTAssertEqual(meta.objects.first?.trueSizeCm, TrueSize(width: 5.5, height: nil))
        XCTAssertNil(meta.objects.first?.facing)
    }
}
