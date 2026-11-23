import SwiftUI

/// Capture tab. Phase 0 shows LiDAR status only; preview, tapping and export arrive in phase 1.
struct CaptureView: View {
    @EnvironmentObject private var captureService: CaptureService

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                LiDARStatusView(availability: captureService.availability)
                Text("Capture is built in phase 1.")
                    .foregroundStyle(.secondary)
            }
            .padding()
            .navigationTitle("Capture")
        }
    }
}

struct LiDARStatusView: View {
    let availability: CaptureService.Availability

    var body: some View {
        switch availability {
        case .available:
            Label("LiDAR camera available", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .unavailable(let reason):
            Label(reason, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
        }
    }
}

#Preview {
    CaptureView().environmentObject(CaptureService())
}
