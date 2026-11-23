import SwiftUI
import UIKit

/// Capture tab: live preview, tap to focus and mark objects, save samples, review and export.
struct CaptureView: View {
    @EnvironmentObject private var captureService: CaptureService
    @EnvironmentObject private var store: SessionStore
    @StateObject private var model = CaptureViewModel()

    @State private var previewProxy = PreviewProxy()
    @State private var showSetup = false
    @State private var showGallery = false
    @State private var showGuidance = false
    @State private var showMeasurements = false
    @State private var exportItem: ExportItem?
    @State private var isExporting = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                if captureService.isLiDARAvailable {
                    cameraArea
                    statusLines
                    controls
                } else {
                    LiDARStatusView(availability: captureService.availability)
                        .padding()
                    Text("Sessions already on this device can still be reviewed and exported.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
            .navigationTitle(store.manifest.map { "Session \($0.sessionId)" } ?? "Capture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarContent }
            .sheet(isPresented: $showSetup) { SessionSetupView().environmentObject(store) }
            .sheet(isPresented: $showGallery) { SessionGalleryView().environmentObject(store) }
            .sheet(isPresented: $showGuidance) { CaptureGuidanceView() }
            .sheet(isPresented: $showMeasurements) { MeasurementsView(measured: $model.measured) }
            .sheet(item: $exportItem) { item in ActivityView(items: [item.url]) }
            .task {
                UIDevice.current.beginGeneratingDeviceOrientationNotifications()
                await captureService.start()
                if captureService.isLiDARAvailable, store.manifest == nil { showSetup = true }
            }
        }
    }

    // MARK: Preview with tap markers

    private var cameraArea: some View {
        ZStack {
            CameraPreviewView(session: captureService.captureSession, proxy: previewProxy) { devicePoint in
                Task { await model.handleTap(devicePoint: devicePoint, service: captureService, store: store) }
            }
            // The square the model will see; taps outside it are refused.
            if let configuration = captureService.configuration {
                SquareGuideView(proxy: previewProxy, imageWidth: configuration.videoWidth,
                                imageHeight: configuration.videoHeight)
            }
            ForEach(Array(model.pendingObjects.enumerated()), id: \.element.id) { index, object in
                if let point = previewProxy.layerPoint(fromDevicePoint: object.devicePoint) {
                    TapMarker(number: index + 1, label: object.label)
                        .position(point)
                        .allowsHitTesting(false)
                }
            }
            if model.isBusy {
                ProgressView().tint(.white)
            }
            if captureService.permissionDenied {
                Text("Camera access is off. Allow it in Settings → UNet-ultra.")
                    .padding()
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
        }
        // The portrait preview of a 4:3 sensor image is 3:4.
        .aspectRatio(3.0 / 4.0, contentMode: .fit)
        .clipped()
    }

    private var statusLines: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 12) {
                Label("LiDAR", systemImage: captureService.isRunning ? "checkmark.circle.fill" : "circle.dotted")
                    .foregroundStyle(captureService.isRunning ? Color.green : Color.secondary)
                Text(String(format: "lens %.3f", captureService.lensPosition))
                Text(model.lastTapDistanceM.map { String(format: "tap %.1f cm", $0 * 100) } ?? "tap – cm")
                Spacer()
            }
            .font(.footnote.monospacedDigit())
            HStack(spacing: 12) {
                Text("\(store.sampleCount) samples, \(store.backgroundOnlyCount) no-object")
                Text("\(model.pendingObjects.count) tapped")
                Spacer()
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
            if let message = model.statusMessage ?? store.errorMessage ?? captureService.errorMessage {
                Text(message).font(.footnote).foregroundStyle(.orange)
            } else if let configuration = captureService.configuration {
                Text(configuration.summary).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal)
    }

    private var controls: some View {
        VStack(spacing: 8) {
            if store.objectClasses.count > 1 {
                Picker("Label for the next tap", selection: $model.selectedClassId) {
                    ForEach(Array(store.objectClasses.enumerated()), id: \.offset) { index, name in
                        Text(name).tag(index + 1)
                    }
                }
                .pickerStyle(.segmented)
            }
            HStack(spacing: 12) {
                Button("Clear taps") { model.clearTaps() }
                    .disabled(model.pendingObjects.isEmpty)
                Button("No object") {
                    Task { await model.capture(backgroundOnly: true, service: captureService, store: store) }
                }
                Button {
                    Task { await model.capture(backgroundOnly: false, service: captureService, store: store) }
                } label: {
                    Label("Capture", systemImage: "camera.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.pendingObjects.isEmpty)
            }
            .buttonStyle(.bordered)
            .disabled(model.isBusy || !captureService.isRunning || store.manifest == nil)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .topBarLeading) {
            Button { showSetup = true } label: { Image(systemName: "folder.badge.plus") }
                .accessibilityLabel("Session")
            Button { showGuidance = true } label: { Image(systemName: "questionmark.circle") }
                .accessibilityLabel("Capture guidance")
        }
        ToolbarItemGroup(placement: .topBarTrailing) {
            Button { showMeasurements = true } label: {
                Image(systemName: model.measured.isEmpty ? "ruler" : "ruler.fill")
            }
            .accessibilityLabel("Measured values")
            Button { showGallery = true } label: { Image(systemName: "photo.on.rectangle") }
                .accessibilityLabel("Gallery")
                .disabled(store.manifest == nil)
            Button {
                Task {
                    isExporting = true
                    if let url = await store.exportCurrentSession() {
                        exportItem = ExportItem(url: url)
                    }
                    isExporting = false
                }
            } label: {
                Image(systemName: "square.and.arrow.up")
            }
            .accessibilityLabel("Export session")
            .disabled(store.sampleCount == 0 || isExporting)
        }
    }
}

/// A numbered marker drawn where an object was tapped.
struct TapMarker: View {
    let number: Int
    let label: String

    var body: some View {
        VStack(spacing: 2) {
            ZStack {
                Circle().stroke(.yellow, lineWidth: 2).frame(width: 28, height: 28)
                Text("\(number)").font(.caption.bold()).foregroundStyle(.yellow)
            }
            Text(label)
                .font(.caption2)
                .padding(.horizontal, 4)
                .background(.black.opacity(0.5), in: Capsule())
                .foregroundStyle(.white)
        }
        // Keep the circle, not the whole stack, centred on the tap.
        .offset(y: 8)
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

/// A finished export, identified so it can drive a sheet.
struct ExportItem: Identifiable {
    let id = UUID()
    let url: URL
}

/// The system share sheet (AirDrop, Files, ...).
struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

#Preview {
    CaptureView()
        .environmentObject(CaptureService())
        .environmentObject(SessionStore())
}
