import SwiftUI

/// Inference tab. Built in phase 6: tap to focus, run the Core ML model, show mask, distance and size.
struct InferenceView: View {
    @EnvironmentObject private var captureService: CaptureService

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                LiDARStatusView(availability: captureService.availability)
                Text("Inference is built in phase 6.")
                    .foregroundStyle(.secondary)
            }
            .padding()
            .navigationTitle("Inference")
        }
    }
}

#Preview {
    InferenceView().environmentObject(CaptureService())
}
