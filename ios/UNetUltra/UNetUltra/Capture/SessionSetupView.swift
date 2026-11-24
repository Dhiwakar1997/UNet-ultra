import SwiftUI
import UIKit

/// Start a new capture session (classes, split) or reopen a stored one.
struct SessionSetupView: View {
    @EnvironmentObject private var store: SessionStore
    @Environment(\.dismiss) private var dismiss

    @State private var classList = "airpods_case"
    @State private var split = SessionSplit.train
    @AppStorage("captureTarget.train") private var trainTarget = CapturePlan.defaultTarget(for: .train)
    @AppStorage("captureTarget.test") private var testTarget = CapturePlan.defaultTarget(for: .test)

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Classes, comma separated", text: $classList)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Picker("Split", selection: $split) {
                        ForEach(SessionSplit.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    Stepper("Target: \(split == .train ? trainTarget : testTarget) samples",
                            value: split == .train ? $trainTarget : $testTarget, in: 10...500, step: 10)
                    Button("Start new session") {
                        store.startSession(
                            classes: classList.split(separator: ",").map(String.init), split: split,
                            deviceModel: CaptureViewModel.deviceModel(),
                            iosVersion: UIDevice.current.systemVersion,
                            appVersion: CaptureViewModel.appVersion())
                        dismiss()
                    }
                } header: {
                    Text("New session")
                } footer: {
                    Text("The guide under the preview spreads the target over 15–70 cm and about 10% no-object shots. "
                         + "The session id is created automatically. \"background\" is always class 0. "
                         + "A test session must come from a different room or day than the training sessions.")
                }

                if let manifest = store.manifest {
                    Section("Current session") {
                        LabeledContent("Id", value: manifest.sessionId)
                        LabeledContent("Split", value: manifest.split.rawValue)
                        LabeledContent("Classes", value: manifest.classes.dropFirst().joined(separator: ", "))
                        LabeledContent("Samples", value: "\(store.sampleCount) (\(store.backgroundOnlyCount) no-object)")
                    }
                }

                if !store.sessionIds.isEmpty {
                    Section("Stored sessions") {
                        ForEach(store.sessionIds, id: \.self) { sessionId in
                            Button(sessionId) {
                                store.openSession(sessionId)
                                dismiss()
                            }
                        }
                    }
                }
            }
            .navigationTitle("Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
    }
}
