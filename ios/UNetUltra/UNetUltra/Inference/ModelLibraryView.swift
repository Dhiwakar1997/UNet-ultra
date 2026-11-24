import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    /// A `.unetmodel` bundle. Declared in UNetUltra-Info.plist; falls back to the bare extension.
    static var unetModelBundle: UTType? {
        UTType("com.unetultra.model") ?? UTType(filenameExtension: ModelContract.bundleExtension)
    }
}

/// Models tab: the library of bundled and imported models, with import, inspect, rename,
/// delete and "use this model".
struct ModelLibraryView: View {
    @EnvironmentObject private var store: ModelStore

    @State private var importKind = ImportKind.bundle
    @State private var showImporter = false

    enum ImportKind {
        case bundle, package

        var contentTypes: [UTType] {
            switch self {
            case .bundle:
                // AirDrop and Files may hand the bundle over as a plain zip or as unknown data.
                let custom: [UTType?] = [UTType.unetModelBundle]
                return custom.compactMap { $0 } + [.zip, .data]
            case .package:
                let custom: [UTType?] = [UTType(filenameExtension: "mlpackage"), UTType(filenameExtension: "mlmodelc")]
                return custom.compactMap { $0 } + [.package, .folder]
            }
        }
    }

    var body: some View {
        NavigationStack {
            List {
                if store.isImporting {
                    Section {
                        HStack(spacing: 12) {
                            ProgressView()
                            Text(store.importStep ?? "Importing…")
                        }
                    }
                }
                if let message = store.message {
                    Section {
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                }
                if store.entries.isEmpty {
                    Section {
                        Text("No models yet. Export a model on the Mac, AirDrop the .unetmodel file to this phone "
                             + "and choose UNet-ultra, or use the import button above.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Section {
                        ForEach(store.entries) { entry in
                            NavigationLink(value: entry.id) {
                                ModelRow(entry: entry, isActive: entry.id == store.activeID)
                            }
                            .swipeActions(edge: .trailing) {
                                if !entry.isBundled {
                                    Button("Delete", role: .destructive) {
                                        Task { await store.delete(entry.id) }
                                    }
                                }
                            }
                            .swipeActions(edge: .leading) {
                                Button("Use") {
                                    Task { await store.activate(entry.id) }
                                }
                                .tint(.accentColor)
                            }
                        }
                    } footer: {
                        Text("Swipe right to use a model, left to delete it. Lens calibration is per phone and is "
                             + "not part of a model; import it on the Inference tab.")
                    }
                }
            }
            .navigationTitle("Models")
            .navigationDestination(for: String.self) { id in
                ModelDetailView(entryID: id)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("Import model bundle (.unetmodel)…") {
                            importKind = .bundle
                            showImporter = true
                        }
                        Button("Import bare .mlpackage folder…") {
                            importKind = .package
                            showImporter = true
                        }
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .accessibilityLabel("Import a model")
                    .disabled(store.isImporting)
                }
            }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: importKind.contentTypes) { outcome in
                switch outcome {
                case .failure(let error):
                    store.message = "Import failed: \(error.localizedDescription)"
                case .success(let url):
                    let kind = importKind
                    Task {
                        switch kind {
                        case .bundle: await store.importBundle(from: url)
                        case .package: await store.importPackage(from: url)
                        }
                    }
                }
            }
            .failedSelfTestConfirmation(store: store)
            .task { await store.loadIfNeeded() }
        }
    }
}

/// Small coloured label for a model's self-test state.
struct SelfTestBadge: View {
    let result: SelfTestResult?
    let hasCard: Bool

    var body: some View {
        if let result {
            Label(result.passed ? "self-test passed" : "self-test failed",
                  systemImage: result.passed ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(result.passed ? Color.green : Color.orange)
        } else {
            Label(hasCard ? "self-test not run" : "no model card", systemImage: "questionmark.circle")
                .foregroundStyle(.secondary)
        }
    }
}

/// One line of the library.
struct ModelRow: View {
    let entry: ModelEntry
    let isActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(entry.displayName).font(.headline).lineLimit(1)
                if isActive {
                    Text("active")
                        .font(.caption2.bold())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.2), in: Capsule())
                }
                if entry.isBundled {
                    Text("bundled").font(.caption2).foregroundStyle(.secondary)
                }
            }
            Text(ModelFormat.headline(entry)).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            if let metrics = ModelFormat.metrics(entry.card?.testMetrics) {
                Text(metrics).font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                SelfTestBadge(result: entry.record.selfTest, hasCard: entry.card != nil)
                if let imported = entry.record.importedAt {
                    Text(imported.formatted(date: .abbreviated, time: .shortened)).foregroundStyle(.secondary)
                }
            }
            .font(.caption2)
        }
        .padding(.vertical, 2)
    }
}

/// Text formatting for cards, shared by the row and the detail screen.
enum ModelFormat {
    static func headline(_ entry: ModelEntry) -> String {
        guard let card = entry.card else { return "No model card: described by the Core ML file itself." }
        var parts: [String] = []
        if let architecture = card.architecture { parts.append(architecture) }
        if let version = card.version { parts.append(version) }
        parts.append("\(card.input.width)×\(card.input.height)")
        if let classes = card.classes { parts.append(classes.dropFirst().joined(separator: ", ")) }
        if let size = card.sizeMb { parts.append(String(format: "%.1f MB", size)) }
        if let precision = card.precision { parts.append(precision) }
        return parts.joined(separator: " · ")
    }

    /// Laptop test metrics from the card, or nil when none were recorded.
    static func metrics(_ metrics: ModelCard.TestMetrics?) -> String? {
        guard let metrics else { return nil }
        var parts: [String] = []
        if let dice = metrics.dice { parts.append(String(format: "Dice %.3f", dice)) }
        if let distance = metrics.distanceErrPct { parts.append(String(format: "distance error %.1f%%", distance)) }
        if let size = metrics.sizeErrPct { parts.append(String(format: "size error %.1f%%", size)) }
        return parts.isEmpty ? nil : "Laptop test: " + parts.joined(separator: " · ")
    }
}

/// Everything the card says about one model, plus rename, self-test and delete.
struct ModelDetailView: View {
    let entryID: String

    @EnvironmentObject private var store: ModelStore
    @Environment(\.dismiss) private var dismiss
    @State private var showRename = false
    @State private var newName = ""
    @State private var confirmDelete = false

    var body: some View {
        Group {
            if let entry = store.entry(entryID) {
                content(entry)
            } else {
                Text("This model is no longer in the library.").foregroundStyle(.secondary)
            }
        }
        .navigationTitle(store.entry(entryID)?.displayName ?? "Model")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func row(_ title: String, _ value: String?) -> some View {
        LabeledContent(title, value: (value?.isEmpty ?? true) ? "–" : (value ?? "–"))
    }

    @ViewBuilder
    private func content(_ entry: ModelEntry) -> some View {
        Form {
            Section {
                Button(entry.id == store.activeID ? "This is the active model" : "Use this model") {
                    Task { await store.activate(entry.id) }
                }
                .disabled(entry.id == store.activeID || store.isLoadingModel)
            }
            Section("Self-test") {
                SelfTestBadge(result: entry.record.selfTest, hasCard: entry.card != nil)
                if let result = entry.record.selfTest {
                    Text(result.summary).font(.footnote)
                }
                if entry.card != nil, !entry.isBundled {
                    Button("Run the self-test again") {
                        Task { await store.rerunSelfTest(entry.id) }
                    }
                    .disabled(store.isImporting)
                }
            }
            if let card = entry.card {
                Section("Model card") {
                    row("Name", card.name)
                    row("Display name", card.displayName)
                    row("Version", card.version)
                    row("Architecture", card.architecture)
                    row("Input", "\(card.input.width)×\(card.input.height), \(card.input.crop ?? CenterSquareCrop.name)")
                    row("Meta features", (card.metaFeatures ?? []).isEmpty ? "none (image only)" : (card.metaFeatures ?? []).joined(separator: ", "))
                    row("Mask channels", card.outputs?.maskLogits?.channels.map { String($0) })
                    row("Depth output", card.hasDepthOutput ? "yes, in \(card.outputs?.depth?.units ?? "m")" : "none: focus distance is used")
                    row("Classes", card.classes?.joined(separator: ", "))
                    row("Mask threshold", card.maskThreshold.map { String(format: "%.2f", $0) })
                    row("Precision", card.precision)
                    row("Size", card.sizeMb.map { String(format: "%.1f MB", $0) })
                    row("Created", card.createdAt)
                    row("Contract version", card.contractVersion.text)
                }
                Section("Laptop test metrics") {
                    row("Dice", card.testMetrics?.dice.map { String(format: "%.3f", $0) })
                    row("Distance error", card.testMetrics?.distanceErrPct.map { String(format: "%.1f%%", $0) })
                    row("Size error", card.testMetrics?.sizeErrPct.map { String(format: "%.1f%%", $0) })
                }
                if let sessions = card.trainedOnSessions, !sessions.isEmpty {
                    Section("Trained on") {
                        ForEach(sessions, id: \.self) { Text($0).font(.footnote) }
                    }
                }
                if let notes = card.notes, !notes.isEmpty {
                    Section("Notes") { Text(notes).font(.footnote) }
                }
            } else {
                Section("Model card") {
                    Text("This model has no card. Its input size and outputs are read from the Core ML file, "
                         + "and it cannot be self-tested.")
                        .font(.footnote)
                }
            }
            Section {
                row("Imported", entry.record.importedAt?.formatted(date: .abbreviated, time: .shortened))
                if !entry.isBundled {
                    Button("Rename…") {
                        newName = entry.displayName
                        showRename = true
                    }
                    Button("Delete this model", role: .destructive) { confirmDelete = true }
                }
            }
        }
        .alert("Display name", isPresented: $showRename) {
            TextField("Name", text: $newName)
            Button("Save") { Task { await store.rename(entry.id, to: newName) } }
            Button("Use the card's name") { Task { await store.rename(entry.id, to: "") } }
            Button("Cancel", role: .cancel) {}
        }
        .confirmationDialog("Delete \(entry.displayName)?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                Task {
                    await store.delete(entry.id)
                    dismiss()
                }
            }
        }
    }
}

/// Asks before a model whose self-test failed is used. Shared by the Inference and Models tabs.
private struct FailedSelfTestConfirmation: ViewModifier {
    @ObservedObject var store: ModelStore

    func body(content: Content) -> some View {
        let pending = store.pendingConfirmationID.flatMap { store.entry($0) }
        return content.alert(
            "Self-test failed",
            isPresented: Binding(get: { store.pendingConfirmationID != nil },
                                 set: { if !$0 { store.pendingConfirmationID = nil } })
        ) {
            Button("Use it anyway", role: .destructive) {
                if let id = store.pendingConfirmationID {
                    Task { await store.activate(id, confirmFailedSelfTest: true) }
                }
            }
            Button("Cancel", role: .cancel) { store.pendingConfirmationID = nil }
        } message: {
            Text("\(pending?.displayName ?? "This model") did not reproduce the laptop's output on this phone. "
                 + (pending?.record.selfTest?.summary ?? "")
                 + " Its numbers here may be wrong.")
        }
    }
}

extension View {
    func failedSelfTestConfirmation(store: ModelStore) -> some View {
        modifier(FailedSelfTestConfirmation(store: store))
    }
}

#Preview {
    ModelLibraryView().environmentObject(ModelStore())
}
