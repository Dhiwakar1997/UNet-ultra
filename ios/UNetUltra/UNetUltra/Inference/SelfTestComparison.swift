import Foundation

/// Outcome of running a bundle's self-test: the model's own output on the phone against the
/// PyTorch output recorded when it was exported.
struct SelfTestResult: Codable, Equatable {
    var passed: Bool
    /// Share of pixels whose class differs, in percent.
    var maskDifferentPercent: Double
    /// Overlap of the two foreground masks, 0 to 1 (1 when both are empty).
    var maskIoU: Double
    /// Depth difference over the expected object's pixels, in centimetres; nil without a depth check.
    var depthMedianDiffCm: Double?
    var depthMaxDiffCm: Double?
    /// Set when the test could not be run at all (missing files, wrong sizes, model error).
    var problem: String?

    static func failure(_ problem: String) -> SelfTestResult {
        SelfTestResult(passed: false, maskDifferentPercent: 100, maskIoU: 0, problem: problem)
    }

    /// One line for the library: what was measured, and what a failure means.
    var summary: String {
        if let problem { return "Self-test could not run: \(problem)" }
        var parts = [String(format: "mask IoU %.3f (%.2f%% of pixels differ)", maskIoU, maskDifferentPercent)]
        if let depthMedianDiffCm, let depthMaxDiffCm {
            parts.append(String(format: "depth differs by %.2f cm median, %.2f cm max", depthMedianDiffCm, depthMaxDiffCm))
        }
        let detail = parts.joined(separator: " · ")
        return passed
            ? "Self-test passed: \(detail)."
            : "Self-test FAILED: \(detail). The phone's output does not match the laptop's, "
                + "which usually means a pre-processing or conversion mismatch."
    }
}

/// The self-test's arithmetic, on plain arrays so it can be unit-tested. Foundation-only.
enum SelfTestComparison {
    static let minimumIoU = 0.95
    static let maximumMedianDepthDiffCm = 1.0

    /// Masks are class ids per pixel (0 = background); depths are metres. All row-major, same length.
    static func compare(predictedMask: [UInt8], expectedMask: [UInt8],
                        predictedDepth: [Float]?, expectedDepth: [Float]?) -> SelfTestResult {
        guard !expectedMask.isEmpty, predictedMask.count == expectedMask.count else {
            return .failure("the model's mask has \(predictedMask.count) pixels, the expected mask \(expectedMask.count).")
        }
        var different = 0, intersection = 0, union = 0
        for index in expectedMask.indices {
            let predicted = predictedMask[index], expected = expectedMask[index]
            if predicted != expected { different += 1 }
            if predicted != 0 && expected != 0 { intersection += 1 }
            if predicted != 0 || expected != 0 { union += 1 }
        }
        let iou = union == 0 ? 1.0 : Double(intersection) / Double(union)
        var result = SelfTestResult(
            passed: iou >= minimumIoU,
            maskDifferentPercent: 100 * Double(different) / Double(expectedMask.count),
            maskIoU: iou)

        if let predictedDepth, let expectedDepth {
            guard predictedDepth.count == expectedMask.count, expectedDepth.count == expectedMask.count else {
                return .failure("the depth maps do not have the mask's size.")
            }
            // Compare where the object is expected; over everything if no object is expected.
            let hasObject = expectedMask.contains { $0 != 0 }
            var differences: [Double] = []
            for index in expectedMask.indices where !hasObject || expectedMask[index] != 0 {
                let difference = abs(Double(predictedDepth[index]) - Double(expectedDepth[index])) * 100
                if difference.isFinite { differences.append(difference) }
            }
            if differences.isEmpty {
                return .failure("the depth output holds no finite values to compare.")
            }
            differences.sort()
            let median = LatencyStats.quantile(sorted: differences, 0.5)
            result.depthMedianDiffCm = median
            result.depthMaxDiffCm = differences[differences.count - 1]
            result.passed = result.passed && median <= maximumMedianDepthDiffCm
        }
        return result
    }
}
