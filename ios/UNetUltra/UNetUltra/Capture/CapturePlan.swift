import Foundation

/// A distance range the plan wants covered, in centimetres (lower inclusive, upper exclusive).
struct DistanceBand: Equatable, Identifiable {
    let lowerCm: Int
    let upperCm: Int

    var id: Int { lowerCm }
    var label: String { "\(lowerCm)–\(upperCm)" }

    func contains(cm: Double) -> Bool { cm >= Double(lowerCm) && cm < Double(upperCm) }
}

/// What one capture session should contain: how many samples, spread over which distances,
/// and how many with no object. Foundation-only, so the guide's logic is unit-testable.
struct CapturePlan: Equatable {
    static let bands = [
        DistanceBand(lowerCm: 15, upperCm: 25), DistanceBand(lowerCm: 25, upperCm: 35),
        DistanceBand(lowerCm: 35, upperCm: 45), DistanceBand(lowerCm: 45, upperCm: 55),
        DistanceBand(lowerCm: 55, upperCm: 70),
    ]
    static let poses = ["lying flat", "standing up", "held in your hand"]

    let split: SessionSplit
    let totalTarget: Int
    /// About 10% of the session, with no object in view.
    let backgroundTarget: Int
    /// Object samples wanted in each of `bands`.
    let bandTargets: [Int]
    /// Test sessions only: how many samples should carry ruler measurements.
    let measuredTarget: Int

    static func defaultTarget(for split: SessionSplit) -> Int { split == .train ? 100 : 20 }

    init(split: SessionSplit, totalTarget: Int) {
        let total = max(totalTarget, 1)
        self.split = split
        self.totalTarget = total
        backgroundTarget = total >= 5 ? max(1, Int((Double(total) * 0.1).rounded())) : 0
        let objects = total - backgroundTarget
        let count = Self.bands.count
        // Split evenly; any remainder goes to the nearer bands, where focus carries most information.
        bandTargets = (0..<count).map { objects / count + ($0 < objects % count ? 1 : 0) }
        measuredTarget = split == .test ? min(5, objects) : 0
    }

    var objectTarget: Int { totalTarget - backgroundTarget }
}

/// Where a session stands against its plan, worked out from the saved samples.
struct CaptureProgress: Equatable {
    let plan: CapturePlan
    let total: Int
    let background: Int
    /// Object samples per distance band, by the LiDAR distance of the focused object.
    let bandCounts: [Int]
    /// Object samples with no LiDAR reading or a distance outside every band.
    let outOfRange: Int
    /// Samples that carry a ruler measurement (size or distance).
    let measured: Int

    init(plan: CapturePlan, samples: [SampleMeta]) {
        var counts = [Int](repeating: 0, count: CapturePlan.bands.count)
        var background = 0, outOfRange = 0, measured = 0
        for sample in samples {
            if sample.backgroundOnly || sample.objects.isEmpty {
                background += 1
                continue
            }
            let index = min(max(sample.focusTapIndex, 0), sample.objects.count - 1)
            let object = sample.objects[index]
            if let band = Self.bandIndex(distanceM: object.lidarDistanceAtTapM) {
                counts[band] += 1
            } else {
                outOfRange += 1
            }
            if sample.objects.contains(where: {
                $0.trueDistanceCm != nil || $0.trueSizeCm.width != nil || $0.trueSizeCm.height != nil
            }) {
                measured += 1
            }
        }
        self.plan = plan
        self.total = samples.count
        self.background = background
        self.bandCounts = counts
        self.outOfRange = outOfRange
        self.measured = measured
    }

    static func bandIndex(distanceM: Double?) -> Int? {
        guard let distanceM, distanceM.isFinite else { return nil }
        return CapturePlan.bands.firstIndex { $0.contains(cm: distanceM * 100) }
    }

    func bandRemaining(_ index: Int) -> Int { max(plan.bandTargets[index] - bandCounts[index], 0) }

    var backgroundRemaining: Int { max(plan.backgroundTarget - background, 0) }

    var isComplete: Bool {
        backgroundRemaining == 0 && CapturePlan.bands.indices.allSatisfy { bandRemaining($0) == 0 }
    }

    /// Share of the plan that is done, 0 to 1. Samples beyond a band's target do not count.
    var fraction: Double {
        let done = CapturePlan.bands.indices.reduce(min(background, plan.backgroundTarget)) {
            $0 + min(bandCounts[$1], plan.bandTargets[$1])
        }
        return min(Double(done) / Double(max(plan.totalTarget, 1)), 1)
    }

    /// Samples that count towards the plan, for the "37 / 100" readout.
    var counted: Int { Int((fraction * Double(plan.totalTarget)).rounded()) }

    /// The band that most needs samples, or nil when every band is full.
    var neediestBand: Int? {
        let open = CapturePlan.bands.indices.filter { bandRemaining($0) > 0 }
        return open.max { bandRemaining($0) < bandRemaining($1) }
    }

    /// One line telling the user what to capture next.
    var nextStep: String {
        if isComplete {
            let ruler = plan.measuredTarget > measured
                ? " Add ruler values to \(plan.measuredTarget - measured) more if you can." : ""
            return "Target reached. Export the session." + ruler
        }
        // A no-object shot is due roughly every tenth sample, and once the object bands are full.
        let objectsDone = bandCounts.reduce(0, +)
        let backgroundDue = backgroundRemaining > 0
            && (neediestBand == nil || (background + 1) * plan.objectTarget <= plan.backgroundTarget * objectsDone)
        if backgroundDue {
            return "Next: a shot with no case in view, then press No object (\(background)/\(plan.backgroundTarget))."
        }
        guard let band = neediestBand else { return "Any distance from 15 to 70 cm." }
        let pose = CapturePlan.poses[bandCounts[band] % CapturePlan.poses.count]
        var line = "Next: \(CapturePlan.bands[band].label) cm, case \(pose) " +
            "(\(bandCounts[band])/\(plan.bandTargets[band]))."
        if plan.measuredTarget > measured {
            line += " Ruler values: \(measured)/\(plan.measuredTarget)."
        }
        return line
    }

    /// Feedback on the tap just made, from its LiDAR distance.
    func tapAdvice(distanceM: Double?) -> String {
        guard let distanceM, distanceM.isFinite else {
            return "No LiDAR reading at the tap. Move back a little or tap the middle of the case."
        }
        let cm = distanceM * 100
        guard let band = Self.bandIndex(distanceM: distanceM) else {
            return String(format: "%.0f cm is outside 15–70 cm. This sample will not count.", cm)
        }
        let label = CapturePlan.bands[band].label
        if bandRemaining(band) > 0 {
            return String(format: "%.0f cm: counts for %@ cm (%d/%d).", cm, label, bandCounts[band], plan.bandTargets[band])
        }
        if let needy = neediestBand {
            return String(format: "%.0f cm: %@ cm is full. Move to %@ cm.", cm, label, CapturePlan.bands[needy].label)
        }
        return String(format: "%.0f cm: %@ cm is full.", cm, label)
    }
}
