import CoreGraphics
import Foundation
import UIKit

/// One tapped object waiting to be saved with the next capture.
struct PendingObject: Identifiable, Equatable {
    let id = UUID()
    /// Tap in normalized capture-device coordinates (sensor orientation).
    var devicePoint: CGPoint
    var classId: Int
    var label: String
    /// LiDAR distance at the tap when it was made, in metres.
    var lidarDistanceM: Double?
}

/// Optional ruler measurements typed in by hand. They describe the object the lens focused on.
struct MeasuredValues: Equatable {
    var widthCm = ""
    var heightCm = ""
    var distanceCm = ""
    var facing = ""
    var notes = ""

    static func number(_ text: String) -> Double? {
        Double(text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: "."))
    }

    var isEmpty: Bool { self == MeasuredValues() }
}

/// Drives the Capture tab: turns taps into focus + pending objects, and a press of Capture
/// into one saved sample. Owns no camera or disk code of its own.
@MainActor
final class CaptureViewModel: ObservableObject {
    @Published private(set) var pendingObjects: [PendingObject] = []
    @Published private(set) var lastFocus: FocusResult?
    @Published private(set) var lastTapDistanceM: Double?
    @Published private(set) var isBusy = false
    @Published var selectedClassId = 1
    @Published var measured = MeasuredValues()
    @Published var statusMessage: String?

    /// Future hook for an on-device mask preview; nothing is installed in phase 1.
    var maskPreviewProvider: MaskPreviewProvider?

    /// Index into `pendingObjects` of the object the lens last focused on.
    private var focusTapIndex = 0

    /// Handle a tap on the preview: focus there, read the LiDAR distance, remember the object.
    func handleTap(devicePoint: CGPoint, service: CaptureService, store: SessionStore) async {
        guard !isBusy, service.isRunning else { return }
        guard (0...1).contains(devicePoint.x), (0...1).contains(devicePoint.y) else { return }  // outside the image
        guard store.manifest != nil else {
            statusMessage = "Start a session first."
            return
        }
        // The model only sees the centre square of the frame, so the object must be inside it.
        // A tap outside neither focuses nor counts as an object.
        if let configuration = service.configuration,
           !CenterSquareCrop.containsDevicePoint(x: Double(devicePoint.x), y: Double(devicePoint.y),
                                                 imageWidth: configuration.videoWidth,
                                                 imageHeight: configuration.videoHeight) {
            statusMessage = CenterSquareCrop.outsideHint
            return
        }
        isBusy = true
        defer { isBusy = false }

        let focus = await service.focus(atDevicePoint: devicePoint)
        lastFocus = focus
        var distance: Double?
        if let frame = await service.currentFrame() {
            distance = frame.lidarDistance(atImagePoint: Self.imagePoint(devicePoint, in: frame))
        }
        lastTapDistanceM = distance

        let classes = store.objectClasses
        let classId = min(max(selectedClassId, 1), max(classes.count, 1))
        let label = classes.indices.contains(classId - 1) ? classes[classId - 1] : "object"
        pendingObjects.append(PendingObject(devicePoint: devicePoint, classId: classId, label: label,
                                            lidarDistanceM: distance))
        focusTapIndex = pendingObjects.count - 1
        statusMessage = focus.settled ? nil : "Focus did not settle; tap again."
    }

    func clearTaps() {
        pendingObjects = []
        focusTapIndex = 0
        lastTapDistanceM = nil
    }

    /// Save one synchronized RGB + depth sample. With `backgroundOnly` the taps are ignored and
    /// the sample is stored with an empty object list.
    func capture(backgroundOnly: Bool, service: CaptureService, store: SessionStore) async {
        guard !isBusy else { return }
        guard store.manifest != nil else {
            statusMessage = "Start a session first."
            return
        }
        guard backgroundOnly || !pendingObjects.isEmpty else {
            statusMessage = "Tap the object first, or use No object."
            return
        }
        guard let frame = await service.currentFrame() else {
            statusMessage = "No camera frame yet."
            return
        }
        isBusy = true
        defer { isBusy = false }

        let objects = backgroundOnly ? [] : annotations(for: frame)
        let meta = SampleMeta(
            sampleId: store.nextSampleId(),
            timestamp: SessionFiles.timestamp(frame.date),
            imageSize: [frame.imageWidth, frame.imageHeight],
            depthSize: [frame.depthWidth, frame.depthHeight],
            intrinsicsImage: frame.intrinsicsImage,
            intrinsicsDepth: frame.intrinsicsDepth,
            intrinsicsReferenceSize: [frame.intrinsicsReferenceWidth, frame.intrinsicsReferenceHeight],
            lensPosition: min(max(frame.lensPosition, 0), 1),
            focusSettled: !frame.isAdjustingFocus && (backgroundOnly || (lastFocus?.settled ?? false)),
            focusTapIndex: backgroundOnly ? 0 : focusTapIndex,
            deviceOrientation: Self.orientationName(UIDevice.current.orientation),
            exposure: Exposure(iso: frame.iso, durationS: frame.exposureDurationSeconds),
            backgroundOnly: backgroundOnly,
            objects: objects,
            notes: measured.notes)

        // Encoding a 1920×1440 JPEG takes a moment; keep it off the main actor.
        let encoded = await Task.detached(priority: .userInitiated) { () -> (Data, Data)? in
            guard let jpeg = ImageEncoding.jpegData(from: frame.image),
                  let png = ImageEncoding.png16Data(values: ImageEncoding.depthMillimetres(from: frame),
                                                    width: frame.depthWidth, height: frame.depthHeight)
            else { return nil }
            return (jpeg, png)
        }.value
        guard let encoded else {
            statusMessage = "Could not encode the frame."
            return
        }
        if store.addSample(meta: meta, imageJPEG: encoded.0, depthPNG: encoded.1) {
            statusMessage = "Saved \(meta.sampleId)" + (backgroundOnly ? " (no object)" : "")
            if !backgroundOnly {
                clearTaps()
                measured.distanceCm = ""  // distance changes shot to shot; size and facing often do not
            }
        }
    }

    /// Pending taps as saved annotations, with tap points in pixels of the saved image.
    private func annotations(for frame: CapturedFrame) -> [ObjectAnnotation] {
        pendingObjects.enumerated().map { index, object in
            let point = Self.imagePoint(object.devicePoint, in: frame)
            var annotation = ObjectAnnotation(
                classId: object.classId, label: object.label,
                tapPointImage: [Double(point.x), Double(point.y)],
                lidarDistanceAtTapM: frame.lidarDistance(atImagePoint: point) ?? object.lidarDistanceM)
            if index == focusTapIndex {
                annotation.trueSizeCm = TrueSize(width: MeasuredValues.number(measured.widthCm),
                                                 height: MeasuredValues.number(measured.heightCm))
                annotation.trueDistanceCm = MeasuredValues.number(measured.distanceCm)
                let facing = measured.facing.trimmingCharacters(in: .whitespaces)
                annotation.facing = facing.isEmpty ? nil : facing
            }
            return annotation
        }
    }

    /// Normalized capture-device point -> pixel coordinates in the frame's (unrotated) image.
    static func imagePoint(_ devicePoint: CGPoint, in frame: CapturedFrame) -> CGPoint {
        let width = Double(frame.imageWidth), height = Double(frame.imageHeight)
        return CGPoint(x: min(max(Double(devicePoint.x) * width, 0), width - 1),
                       y: min(max(Double(devicePoint.y) * height, 0), height - 1))
    }

    static func orientationName(_ orientation: UIDeviceOrientation) -> String {
        switch orientation {
        case .portrait: return "portrait"
        case .portraitUpsideDown: return "portraitUpsideDown"
        case .landscapeLeft: return "landscapeLeft"
        case .landscapeRight: return "landscapeRight"
        case .faceUp: return "faceUp"
        case .faceDown: return "faceDown"
        default: return "unknown"
        }
    }

    /// Hardware identifier such as "iPhone15,3".
    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { buffer in
            String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    static func appVersion() -> String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
    }
}
