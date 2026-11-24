import XCTest
@testable import UNetUltra

final class CapturePlanTests: XCTestCase {
    private func sample(_ id: Int, distanceM: Double?, background: Bool = false, measuredCm: Double? = nil) -> SampleMeta {
        var objects: [ObjectAnnotation] = []
        if !background {
            var object = ObjectAnnotation(classId: 1, label: "airpods_case", tapPointImage: [960, 720],
                                          lidarDistanceAtTapM: distanceM)
            object.trueDistanceCm = measuredCm
            objects = [object]
        }
        return SampleMeta(
            sampleId: String(format: "%04d", id), timestamp: "2026-01-01T12:00:00Z",
            imageSize: [1920, 1440], depthSize: [256, 192],
            intrinsicsImage: Intrinsics(fx: 1450, fy: 1450, cx: 960, cy: 720),
            intrinsicsDepth: Intrinsics(fx: 193, fy: 193, cx: 128, cy: 96),
            intrinsicsReferenceSize: [4032, 3024], lensPosition: 0.5, focusSettled: true, focusTapIndex: 0,
            deviceOrientation: "portrait", exposure: Exposure(iso: 100, durationS: 0.01),
            backgroundOnly: background, objects: objects)
    }

    func testTrainPlanOfOneHundred() {
        let plan = CapturePlan(split: .train, totalTarget: 100)
        XCTAssertEqual(plan.backgroundTarget, 10)
        XCTAssertEqual(plan.bandTargets, [18, 18, 18, 18, 18])
        XCTAssertEqual(plan.measuredTarget, 0)
        XCTAssertEqual(plan.bandTargets.reduce(0, +) + plan.backgroundTarget, 100)
    }

    func testTestPlanOfTwenty() {
        let plan = CapturePlan(split: .test, totalTarget: 20)
        XCTAssertEqual(plan.backgroundTarget, 2)
        XCTAssertEqual(plan.bandTargets, [4, 4, 4, 3, 3])
        XCTAssertEqual(plan.measuredTarget, 5)
    }

    func testProgressCountsBandsBackgroundAndOutOfRange() {
        let plan = CapturePlan(split: .test, totalTarget: 20)
        let samples = [
            sample(1, distanceM: 0.20), sample(2, distanceM: 0.249), sample(3, distanceM: 0.25),
            sample(4, distanceM: 0.62, measuredCm: 62), sample(5, distanceM: nil), sample(6, distanceM: 0.9),
            sample(7, distanceM: nil, background: true),
        ]
        let progress = CaptureProgress(plan: plan, samples: samples)
        XCTAssertEqual(progress.bandCounts, [2, 1, 0, 0, 1])
        XCTAssertEqual(progress.background, 1)
        XCTAssertEqual(progress.outOfRange, 2)
        XCTAssertEqual(progress.measured, 1)
        XCTAssertEqual(progress.counted, 5)
        XCTAssertFalse(progress.isComplete)
        XCTAssertEqual(progress.neediestBand, 2)
        XCTAssertTrue(progress.nextStep.contains("35–45"))
    }

    func testSamplesBeyondABandsTargetDoNotCount() {
        let plan = CapturePlan(split: .test, totalTarget: 20)
        let samples = (1...9).map { sample($0, distanceM: 0.2) }
        let progress = CaptureProgress(plan: plan, samples: samples)
        XCTAssertEqual(progress.bandCounts[0], 9)
        XCTAssertEqual(progress.counted, 4)
        XCTAssertTrue(progress.tapAdvice(distanceM: 0.2).contains("full"))
        XCTAssertTrue(progress.tapAdvice(distanceM: 0.4).contains("counts for 35–45"))
        XCTAssertTrue(progress.tapAdvice(distanceM: 0.05).contains("outside"))
        XCTAssertTrue(progress.tapAdvice(distanceM: nil).contains("No LiDAR"))
    }

    func testCompleteSessionSaysExport() {
        let plan = CapturePlan(split: .train, totalTarget: 10)
        XCTAssertEqual(plan.bandTargets, [2, 2, 2, 2, 1])
        var samples: [SampleMeta] = []
        for (band, distance) in [0.2, 0.3, 0.4, 0.5, 0.6].enumerated() {
            for copy in 0..<plan.bandTargets[band] {
                samples.append(sample(band * 10 + copy, distanceM: distance))
            }
        }
        var progress = CaptureProgress(plan: plan, samples: samples)
        XCTAssertFalse(progress.isComplete)
        XCTAssertTrue(progress.nextStep.contains("No object"))
        samples.append(sample(99, distanceM: nil, background: true))
        progress = CaptureProgress(plan: plan, samples: samples)
        XCTAssertTrue(progress.isComplete)
        XCTAssertEqual(progress.fraction, 1)
        XCTAssertTrue(progress.nextStep.contains("Export"))
    }

    func testANoObjectShotIsPromptedPartWayThrough() {
        let plan = CapturePlan(split: .train, totalTarget: 100)
        let few = (1...5).map { sample($0, distanceM: 0.2) }
        XCTAssertFalse(CaptureProgress(plan: plan, samples: few).nextStep.contains("No object"))
        let more = (1...14).map { sample($0, distanceM: 0.2 + Double($0 % 5) * 0.1) }
        XCTAssertTrue(CaptureProgress(plan: plan, samples: more).nextStep.contains("No object"))
    }
}
