import SwiftUI

/// The live capture guide under the preview: progress against the session's plan, how many
/// samples each distance band has, and what to capture next.
struct CaptureGuideView: View {
    let progress: CaptureProgress
    /// Advice about the tap that is waiting to be captured, if there is one.
    let tapAdvice: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: progress.fraction)
                    .tint(progress.isComplete ? Color.green : Color.accentColor)
                Text("\(progress.counted) / \(progress.plan.totalTarget)")
                    .font(.footnote.monospacedDigit().bold())
            }
            HStack(spacing: 4) {
                ForEach(Array(CapturePlan.bands.enumerated()), id: \.element.id) { index, band in
                    GuideChip(title: band.label, count: progress.bandCounts[index],
                              target: progress.plan.bandTargets[index],
                              highlighted: progress.neediestBand == index)
                }
                GuideChip(title: "none", count: progress.background, target: progress.plan.backgroundTarget,
                          highlighted: false)
            }
            Text(tapAdvice ?? progress.nextStep)
                .font(.footnote)
                .foregroundStyle(progress.isComplete && tapAdvice == nil ? Color.green : Color.primary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal)
    }
}

/// One cell of the guide: a distance band (in cm) or "none", with captured / wanted.
private struct GuideChip: View {
    let title: String
    let count: Int
    let target: Int
    let highlighted: Bool

    private var isFull: Bool { count >= target }

    var body: some View {
        VStack(spacing: 0) {
            Text(title).font(.caption2)
            Text("\(count)/\(target)").font(.caption2.monospacedDigit().bold())
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 3)
        .background(isFull ? Color.green.opacity(0.25) : Color.secondary.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6)
            .stroke(highlighted ? Color.accentColor : Color.clear, lineWidth: 1.5))
    }
}
