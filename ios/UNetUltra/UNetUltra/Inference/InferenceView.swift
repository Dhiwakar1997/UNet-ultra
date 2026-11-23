import SwiftUI
import UniformTypeIdentifiers

/// Inference tab: tap the object on the live preview and see the model's mask, distance and
/// size, with the LiDAR reading beside them for comparison.
struct InferenceView: View {
    @EnvironmentObject private var captureService: CaptureService
    @StateObject private var assets = ModelAssets()
    @StateObject private var model = InferenceViewModel()

    @State private var previewProxy = PreviewProxy()
    @State private var importKind = ImportKind.calibration
    @State private var showImporter = false
    @State private var showMeasurements = false
    @State private var exportItem: ExportItem?

    enum ImportKind {
        case model, calibration

        var contentTypes: [UTType] {
            switch self {
            case .calibration:
                return [.json]
            case .model:
                let custom: [UTType?] = [UTType(filenameExtension: "mlpackage"), UTType(filenameExtension: "mlmodelc")]
                return custom.compactMap { $0 } + [.package, .folder]
            }
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                if captureService.isLiDARAvailable {
                    cameraArea
                } else {
                    LiDARStatusView(availability: captureService.availability)
                        .padding()
                }
                assetStatus
                if captureService.isLiDARAvailable {
                    readout
                    controls
                } else {
                    Spacer()
                }
            }
            .navigationTitle(assets.model?.descriptor.displayName ?? "Inference")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarContent }
            .sheet(isPresented: $showMeasurements) { MeasurementsView(measured: $model.measured) }
            .sheet(item: $exportItem) { item in ActivityView(items: [item.url]) }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: importKind.contentTypes) { outcome in
                handleImport(outcome)
            }
            .task {
                await assets.loadIfNeeded()
                await captureService.start()
            }
        }
    }

    private func handleImport(_ outcome: Result<URL, Error>) {
        switch outcome {
        case .failure(let error):
            assets.message = "Import failed: \(error.localizedDescription)"
        case .success(let url):
            switch importKind {
            case .calibration: assets.importCalibration(from: url)
            case .model: Task { await assets.importModel(from: url) }
            }
        }
    }

    // MARK: Preview with the mask overlay

    private var cameraArea: some View {
        ZStack {
            CameraPreviewView(session: captureService.captureSession, proxy: previewProxy) { devicePoint in
                Task { await model.handleTap(devicePoint: devicePoint, service: captureService, assets: assets) }
            }
            if let configuration = captureService.configuration {
                SquareGuideView(proxy: previewProxy, imageWidth: configuration.videoWidth,
                                imageHeight: configuration.videoHeight)
            }
            if let overlay = model.overlay, let result = model.result {
                MaskOverlayView(image: overlay, crop: result.crop, proxy: previewProxy)
                if let centroid = result.centroid,
                   let point = previewProxy.layerPoint(fromDevicePoint: CGPoint(x: centroid.x, y: centroid.y)) {
                    Text(result.classLabel)
                        .font(.caption.bold())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.black.opacity(0.6), in: Capsule())
                        .foregroundStyle(.white)
                        .position(point)
                        .allowsHitTesting(false)
                }
            }
            if let tap = model.activeTap, let point = previewProxy.layerPoint(fromDevicePoint: tap) {
                Circle().stroke(.yellow, lineWidth: 2).frame(width: 28, height: 28)
                    .position(point)
                    .allowsHitTesting(false)
            }
            if model.isBusy || assets.isLoadingModel {
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

    // MARK: Model and calibration status

    private var assetStatus: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(assets.modelStatus, systemImage: assets.model == nil ? "exclamationmark.triangle" : "cube")
                .foregroundStyle(assets.model == nil ? Color.orange : Color.secondary)
            Label(assets.calibrationStatus,
                  systemImage: assets.calibration == nil ? "exclamationmark.triangle" : "scope")
                .foregroundStyle(assets.calibration == nil ? Color.orange : Color.secondary)
            if let message = model.statusMessage ?? assets.message ?? captureService.errorMessage {
                Text(message).foregroundStyle(.orange)
            }
        }
        .font(.caption)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal)
    }

    // MARK: Numbers

    private static func cm(_ metres: Double?, digits: Int = 1) -> String {
        guard let metres, metres.isFinite else { return "–" }
        return String(format: "%.\(digits)f cm", metres * 100)
    }

    private var focusDistance: Double? {
        assets.calibration?.focusDistance(lensPosition: captureService.lensPosition)
    }

    private var readout: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let result = model.result, result.found {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model: \(result.modelDisplayName) · \(result.classLabel)")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 16) {
                        VStack(alignment: .leading) {
                            Text("Distance").font(.caption2).foregroundStyle(.secondary)
                            Text(Self.cm(result.measurement?.distanceM)).font(.title3.monospacedDigit().bold())
                        }
                        VStack(alignment: .leading) {
                            Text("Width × height").font(.caption2).foregroundStyle(.secondary)
                            Text(sizeText(result.measurement)).font(.title3.monospacedDigit().bold())
                        }
                    }
                    if result.depthSource == .focusDistance {
                        Text("No depth output: distance is the calibrated focus distance.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("LiDAR reference (not used by the model)")
                        .font(.caption2).foregroundStyle(.secondary)
                    HStack(spacing: 16) {
                        Text(Self.cm(result.lidarDistanceM)).font(.body.monospacedDigit())
                        Text("difference \(differenceText(result.distanceDifferenceM))")
                            .font(.body.monospacedDigit())
                    }
                }
                Text(String(format: "latency %.0f ms · mask %d px · lens %.3f · focus %@",
                            result.latencyMs, result.areaPx, result.lensPosition, Self.cm(result.focusDistanceM)))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            } else {
                Text("Tap the object inside the square.")
                    .font(.callout).foregroundStyle(.secondary)
                Text(String(format: "lens %.3f · focus %@", captureService.lensPosition, Self.cm(focusDistance)))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal)
    }

    private func sizeText(_ measurement: ObjectMeasurement?) -> String {
        guard let measurement else { return "–" }
        return String(format: "%.1f × %.1f cm", measurement.widthM * 100, measurement.heightM * 100)
    }

    private func differenceText(_ metres: Double?) -> String {
        guard let metres, metres.isFinite else { return "–" }
        return String(format: "%+.1f cm", metres * 100)
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button("Clear") { model.clearResult() }
                .disabled(model.result == nil)
            Button {
                model.logResult()
            } label: {
                Label(model.resultLogged ? "Logged" : "Log result", systemImage: "square.and.pencil")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!(model.result?.found ?? false) || model.resultLogged)
            Text("\(model.logRowCount) logged")
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.bordered)
        .disabled(model.isBusy)
        .padding(.horizontal)
        .padding(.bottom, 8)
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Menu {
                Button("Import model (.mlpackage)…") {
                    importKind = .model
                    showImporter = true
                }
                Button("Import calibration.json…") {
                    importKind = .calibration
                    showImporter = true
                }
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .accessibilityLabel("Import model or calibration")
            .disabled(assets.isLoadingModel)
        }
        ToolbarItemGroup(placement: .topBarTrailing) {
            Button { showMeasurements = true } label: {
                Image(systemName: model.measured.isEmpty ? "ruler" : "ruler.fill")
            }
            .accessibilityLabel("Measured values")
            Button {
                if let url = model.logFileForExport() { exportItem = ExportItem(url: url) }
            } label: {
                Image(systemName: "square.and.arrow.up")
            }
            .accessibilityLabel("Export results CSV")
            .disabled(model.logRowCount == 0)
        }
    }
}

#Preview {
    InferenceView().environmentObject(CaptureService())
}
