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
    /// Live mode: the model runs continuously and the lens follows the object it finds.
    @Published private(set) var isLive = false
    /// Model runs per second in live mode.
    @Published private(set) var liveRate: Double = 0

    private var liveTask: Task<Void, Never>?
    /// Where the followed object is, in normalized capture-device coordinates.
    private var liveSelection: CGPoint?

    let log: ResultLog

    init() {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        // An older log with fewer columns is left alone; new rows then go to inference_log_v2.csv.
        log = ResultLog.resolve(directory: documents, baseName: "inference_log", header: InferenceLogRow.header)
        logRowCount = log.rowCount()
    }

    /// Tap on the preview: focus there, wait for the lens, then run the model on the next frame.
    func handleTap(devicePoint: CGPoint, service: CaptureService, store: ModelStore, assets: ModelAssets) async {
        guard !isBusy, service.isRunning else { return }
        guard (0...1).contains(devicePoint.x), (0...1).contains(devicePoint.y) else { return }
        if let configuration = service.configuration,
           !CenterSquareCrop.containsDevicePoint(x: Double(devicePoint.x), y: Double(devicePoint.y),
                                                 imageWidth: configuration.videoWidth,
                                                 imageHeight: configuration.videoHeight) {
            statusMessage = CenterSquareCrop.outsideHint
            return
        }
        guard !store.isLoadingModel else {
            statusMessage = "The model is still loading."
            return
        }
        guard let segmentation = store.activeModel else {
            statusMessage = ModelStore.missingModelHint
            return
        }
        // Remember which model this run belongs to, in case the user switches while it runs.
        let entry = store.activeEntry
        let units = store.computeUnits
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
            let measured = try await Task.detached(priority: .userInitiated) {
                try InferenceRunner.run(model: segmentation, calibration: calibration, frame: frame, tapImagePoint: tap)
            }.value
            var outcome = measured
            outcome.modelVersion = entry?.version ?? ""
            outcome.architecture = entry?.architecture ?? ""
            outcome.computeUnits = units.rawValue
            if store.activeID == entry?.id, store.computeUnits == units {
                store.recordLatency(outcome.latencyMs)
            }
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

    // MARK: Live mode

    /// Run the model continuously on the newest frame. Each result's mask centre becomes the
    /// autofocus point, so the lens (and with it the focus input) follows the object without a tap.
    func startLive(service: CaptureService, store: ModelStore, assets: ModelAssets) {
        guard liveTask == nil else { return }
        isLive = true
        liveSelection = nil
        liveTask = Task { [weak self] in
            var lastFramePresentation = -1.0
            var lastFocusPoint: CGPoint?
            var misses = 0
            var lastRun = Date()
            while !Task.isCancelled {
                guard let self else { return }
                guard service.isRunning, !store.isLoadingModel, let segmentation = store.activeModel,
                      let frame = await service.currentFrame(), frame.presentationTime != lastFramePresentation
                else {
                    try? await Task.sleep(nanoseconds: 40_000_000)
                    continue
                }
                lastFramePresentation = frame.presentationTime
                let entry = store.activeEntry
                let units = store.computeUnits
                let calibration = assets.calibration
                let selection = self.liveSelection.map { CaptureViewModel.imagePoint($0, in: frame) }
                do {
                    let measured = try await Task.detached(priority: .userInitiated) {
                        try InferenceRunner.run(model: segmentation, calibration: calibration, frame: frame,
                                                tapImagePoint: selection)
                    }.value
                    if Task.isCancelled { break }
                    var outcome = measured
                    outcome.modelVersion = entry?.version ?? ""
                    outcome.architecture = entry?.architecture ?? ""
                    outcome.computeUnits = units.rawValue
                    if store.activeID == entry?.id, store.computeUnits == units {
                        store.recordLatency(outcome.latencyMs)
                    }
                    self.result = outcome
                    self.resultLogged = false
                    self.overlay = outcome.found
                        ? MaskImage.make(pixels: outcome.pixels, width: outcome.maskWidth, height: outcome.maskHeight)
                        : nil
                    if outcome.found, let centroid = outcome.centroid {
                        misses = 0
                        let point = CGPoint(x: centroid.x, y: centroid.y)
                        self.liveSelection = point
                        // Re-aim the autofocus only when the object has moved noticeably.
                        let moved = lastFocusPoint.map { hypot($0.x - point.x, $0.y - point.y) > 0.04 } ?? true
                        if moved {
                            service.trackFocus(atDevicePoint: point)
                            lastFocusPoint = point
                        }
                        self.statusMessage = frame.isAdjustingFocus ? "Focusing… the distance settles when the lens stops." : nil
                    } else {
                        misses += 1
                        if misses >= 5 {
                            self.liveSelection = nil
                            lastFocusPoint = nil
                        }
                        self.statusMessage = "Looking for the object…"
                    }
                } catch {
                    self.statusMessage = error.localizedDescription
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
                let now = Date()
                let interval = now.timeIntervalSince(lastRun)
                lastRun = now
                if interval > 0 { self.liveRate = 0.8 * self.liveRate + 0.2 / interval }
                await Task.yield()
            }
        }
    }

    func stopLive() {
        liveTask?.cancel()
        liveTask = nil
        isLive = false
        liveRate = 0
    }

    /// A tap while live: follow the object at that point (useful when there are several).
    func liveTap(devicePoint: CGPoint, service: CaptureService) {
        guard (0...1).contains(devicePoint.x), (0...1).contains(devicePoint.y) else { return }
        if let configuration = service.configuration,
           !CenterSquareCrop.containsDevicePoint(x: Double(devicePoint.x), y: Double(devicePoint.y),
                                                 imageWidth: configuration.videoWidth,
                                                 imageHeight: configuration.videoHeight) {
            statusMessage = CenterSquareCrop.outsideHint
            return
        }
        liveSelection = devicePoint
        service.trackFocus(atDevicePoint: devicePoint)
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
