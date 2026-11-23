import CoreGraphics
import Foundation

/// Drives the Inference tab: a tap focuses, the model runs on the frame that follows, and the
/// result can be appended to a CSV log. Owns no camera or Core ML code of its own.
@MainActor
final class InferenceViewModel: ObservableObject {
    @Published private(set) var result: InferenceResult?
    @Published private(set) var overlay: CGImage?
    @Published private(set) var isBusy = false
    /// The tap being processed, in normalized capture-device coordinates.
    @Published private(set) var activeTap: CGPoint?
    @Published private(set) var logRowCount = 0
    @Published private(set) var resultLogged = false
    @Published var statusMessage: String?
    @Published var measured = MeasuredValues()

    let log: ResultLog

    init() {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        log = ResultLog(fileURL: documents.appendingPathComponent("inference_log.csv"), header: InferenceLogRow.header)
        logRowCount = log.rowCount()
    }

    /// Tap on the preview: focus there, wait for the lens, then run the model on the next frame.
    func handleTap(devicePoint: CGPoint, service: CaptureService, assets: ModelAssets) async {
        guard !isBusy, service.isRunning else { return }
        guard (0...1).contains(devicePoint.x), (0...1).contains(devicePoint.y) else { return }
        if let configuration = service.configuration,
           !CenterSquareCrop.containsDevicePoint(x: Double(devicePoint.x), y: Double(devicePoint.y),
                                                 imageWidth: configuration.videoWidth,
                                                 imageHeight: configuration.videoHeight) {
            statusMessage = CenterSquareCrop.outsideHint
            return
        }
        guard let segmentation = assets.model else {
            statusMessage = ModelAssets.missingModelHint
            return
        }
        let calibration = assets.calibration
        isBusy = true
        activeTap = devicePoint
        defer {
            isBusy = false
            activeTap = nil
        }

        let focus = await service.focus(atDevicePoint: devicePoint)
        guard let frame = await service.currentFrame() else {
            statusMessage = "No camera frame yet."
            return
        }
        let tap = CaptureViewModel.imagePoint(devicePoint, in: frame)
        do {
            let outcome = try await Task.detached(priority: .userInitiated) {
                try InferenceRunner.run(model: segmentation, calibration: calibration, frame: frame, tapImagePoint: tap)
            }.value
            result = outcome
            resultLogged = false
            overlay = outcome.found
                ? MaskImage.make(pixels: outcome.pixels, width: outcome.maskWidth, height: outcome.maskHeight) : nil
            if !outcome.found {
                statusMessage = "The model found no object in this frame."
            } else if outcome.measurement == nil {
                statusMessage = "This model has no depth output, so it needs calibration.json to give a distance."
            } else if !focus.settled {
                statusMessage = "Focus did not settle; the focus input may be off. Tap again."
            } else if !outcome.containsTap {
                statusMessage = "Nothing under the tap; showing the largest object."
            } else {
                statusMessage = nil
            }
        } catch {
            result = nil
            overlay = nil
            statusMessage = error.localizedDescription
        }
    }

    func clearResult() {
        result = nil
        overlay = nil
        statusMessage = nil
    }

    /// Append the current result, with any hand-measured values, to the CSV log.
    func logResult() {
        guard let result, result.found else { return }
        let row = InferenceLogRow(
            timestamp: SessionFiles.timestamp(), deviceModel: CaptureViewModel.deviceModel(), result: result,
            measuredWidthCm: MeasuredValues.number(measured.widthCm),
            measuredHeightCm: MeasuredValues.number(measured.heightCm),
            measuredDistanceCm: MeasuredValues.number(measured.distanceCm),
            notes: measured.notes)
        do {
            try log.append(row.fields)
            logRowCount = log.rowCount()
            resultLogged = true
            statusMessage = "Logged (\(logRowCount) rows)."
        } catch {
            statusMessage = "Could not write the log: \(error.localizedDescription)"
        }
    }

    /// The log file if it has any rows, for the share sheet.
    func logFileForExport() -> URL? {
        logRowCount > 0 && FileManager.default.fileExists(atPath: log.fileURL.path) ? log.fileURL : nil
    }
}
