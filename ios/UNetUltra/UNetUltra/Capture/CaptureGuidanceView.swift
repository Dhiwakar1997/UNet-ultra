import SwiftUI

/// The capture checklist, and the optional hand-measured values.
struct CaptureGuidanceView: View {
    @Environment(\.dismiss) private var dismiss

    private struct Topic: Identifiable {
        let title: String
        let items: [String]
        var id: String { title }
    }

    private let sections: [Topic] = [
        Topic(title: "The object", items: [
            "Case closed, always in the same cover.",
            "Tap the case to focus on it, wait for the marker, then press Capture.",
            "Keep the whole case inside the square: the model only sees that part of the frame.",
        ]),
        Topic(title: "Distance and pose", items: [
            "15–70 cm from the camera; vary the distance across the whole range.",
            "Vary the angle and which face of the case is visible.",
            "Lying flat, standing up, and held in the hand.",
        ]),
        Topic(title: "Surroundings", items: [
            "Varied surfaces and backgrounds.",
            "Varied lighting: daylight, lamp light, dim, with shadows.",
        ]),
        Topic(title: "Background-only shots", items: [
            "About 1 in 10 samples with no case in view: press No object.",
            "Include scenes that look like where the case usually sits.",
        ]),
        Topic(title: "The guide under the preview", items: [
            "It shows how many samples each distance band has and what to capture next.",
            "Default plan: 100 samples for a training session, 20 for a test session.",
            "A sample counts for the band its LiDAR distance falls in; extra samples in a full band do not count.",
        ]),
        Topic(title: "Sessions", items: [
            "Each session is entirely train or test.",
            "Capture the test session in a different room or on a different day.",
        ]),
    ]

    var body: some View {
        NavigationStack {
            List {
                ForEach(sections) { section in
                    Section(section.title) {
                        ForEach(section.items, id: \.self) { Text($0) }
                    }
                }
            }
            .navigationTitle("Capture guidance")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
    }
}

/// Optional values measured with a ruler or tape; saved with the object the lens focused on.
struct MeasurementsView: View {
    @Binding var measured: MeasuredValues
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Width (cm)", text: $measured.widthCm).keyboardType(.decimalPad)
                    TextField("Height (cm)", text: $measured.heightCm).keyboardType(.decimalPad)
                    TextField("Facing, e.g. front face to camera", text: $measured.facing)
                } header: {
                    Text("Size as seen from the camera")
                } footer: {
                    Text("Kept between captures until you change them.")
                }
                Section {
                    TextField("Distance (cm)", text: $measured.distanceCm).keyboardType(.decimalPad)
                } header: {
                    Text("Camera to object")
                } footer: {
                    Text("Cleared after each capture.")
                }
                Section("Notes") {
                    TextField("Notes for the next samples", text: $measured.notes)
                }
                Button("Clear all", role: .destructive) { measured = MeasuredValues() }
            }
            .navigationTitle("Measured values")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
