import AVFoundation
import CoreVideo
import Foundation
import simd

enum CaptureError: Error, LocalizedError {
    case noLiDARCamera
    case noSuitableFormat
    case cannotConfigure(String)

    var errorDescription: String? {
        switch self {
        case .noLiDARCamera: return "This device has no LiDAR depth camera."
        case .noSuitableFormat: return "No 1920×1440 video format with LiDAR depth was found."
        case .cannotConfigure(let reason): return "The camera could not be configured: \(reason)"
        }
    }
}

/// Owns the AVFoundation capture session: the LiDAR depth camera delivering synchronized
/// video + depth frames. One instance is shared by the Capture and Inference tabs, so both
/// see frames that went through exactly the same pipeline.
///
/// AVFoundation is used instead of ARKit so that the lens is free to focus and `lensPosition`
/// carries information.
@MainActor
final class CaptureService: ObservableObject {
    enum Availability: Equatable {
        case available
        case unavailable(reason: String)
    }

    @Published private(set) var availability: Availability
    @Published private(set) var isRunning = false
    @Published private(set) var permissionDenied = false
    /// Current `lensPosition` (0 nearest to 1 furthest), refreshed a few times a second.
    @Published private(set) var lensPosition: Double = 0
    @Published private(set) var configuration: CaptureConfiguration?
    @Published private(set) var errorMessage: String?

    private let pipeline = CapturePipeline()
    private var lensTask: Task<Void, Never>?

    init() {
        availability = Self.detectAvailability()
    }

    var isLiDARAvailable: Bool { availability == .available }

    /// The session behind the live preview layer.
    var captureSession: AVCaptureSession { pipeline.session }

    /// The 1x wide camera paired with the LiDAR scanner, or nil on devices without one.
    nonisolated static func lidarDevice() -> AVCaptureDevice? {
        #if targetEnvironment(simulator)
        return nil
        #else
        return AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
        #endif
    }

    private static func detectAvailability() -> Availability {
        #if targetEnvironment(simulator)
        return .unavailable(reason: "The simulator has no camera. Run on an iPhone Pro with LiDAR.")
        #else
        if lidarDevice() != nil {
            return .available
        }
        return .unavailable(reason: "This device has no LiDAR depth camera.")
        #endif
    }

    /// Ask for camera permission if needed, then configure and start the session. Safe to call again.
    func start() async {
        guard isLiDARAvailable, !isRunning else { return }
        guard await Self.requestCameraAccess() else {
            permissionDenied = true
            return
        }
        permissionDenied = false
        do {
            configuration = try await pipeline.start()
            errorMessage = nil
            isRunning = true
            startLensUpdates()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Stop the session, for example when the app goes to the background.
    func stop() async {
        guard isRunning else { return }
        lensTask?.cancel()
        lensTask = nil
        await pipeline.stop()
        isRunning = false
    }

    /// The most recent synchronized frame, or nil if none has arrived yet.
    func currentFrame() async -> CapturedFrame? {
        pipeline.latestFrame()
    }

    /// Focus (and meter exposure) at a point in normalized capture-device coordinates: (0, 0)
    /// is the top-left and (1, 1) the bottom-right of the sensor-orientation image. Returns
    /// once the lens stops moving, or after a timeout with `settled == false`.
    func focus(atDevicePoint point: CGPoint) async -> FocusResult {
        let result = await pipeline.focus(atDevicePoint: point)
        lensPosition = result.lensPosition
        return result
    }

    /// LiDAR distance in metres at a point in image pixels of `frame` (median of a small window).
    nonisolated static func lidarDistance(in frame: CapturedFrame, atImagePoint point: CGPoint) -> Double? {
        frame.lidarDistance(atImagePoint: point)
    }

    private static func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    private func startLensUpdates() {
        lensTask?.cancel()
        lensTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                // Publish only real changes, so the views are not redrawn for nothing.
                if let position = self.pipeline.currentLensPosition(), abs(position - self.lensPosition) > 0.0005 {
                    self.lensPosition = position
                }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
    }
}

/// The AVFoundation side of `CaptureService`. Everything here runs off the main actor: the
/// session is configured on `sessionQueue` and frames arrive on `dataQueue`.
final class CapturePipeline: NSObject, AVCaptureDataOutputSynchronizerDelegate, @unchecked Sendable {
    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "unetultra.capture.session")
    private let dataQueue = DispatchQueue(label: "unetultra.capture.data")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private var device: AVCaptureDevice?
    private var configuration: CaptureConfiguration?

    private let lock = NSLock()
    private var storedFrame: CapturedFrame?

    static let videoWidth: Int32 = 1920
    static let videoHeight: Int32 = 1440
    static let focusTimeoutSeconds = 2.0

    // MARK: Session lifecycle

    func start() async throws -> CaptureConfiguration {
        try await withCheckedThrowingContinuation { continuation in
            sessionQueue.async {
                do {
                    let configuration = try self.configureIfNeeded()
                    if !self.session.isRunning {
                        self.session.startRunning()
                    }
                    continuation.resume(returning: configuration)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func stop() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            sessionQueue.async {
                if self.session.isRunning {
                    self.session.stopRunning()
                }
                continuation.resume()
            }
        }
    }

    /// Build the session once: LiDAR camera at 1920×1440, 1× zoom, BGRA video + Float32 depth,
    /// delivered together by a synchronizer. Called on `sessionQueue`.
    private func configureIfNeeded() throws -> CaptureConfiguration {
        if let existing = self.configuration { return existing }
        guard let device = CaptureService.lidarDevice() else { throw CaptureError.noLiDARCamera }

        // A 4:3 1920×1440 format that can deliver depth; prefer full-range 4:2:0 if there are several.
        let candidates = device.formats.filter { format in
            let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return size.width == Self.videoWidth && size.height == Self.videoHeight
                && !format.supportedDepthDataFormats.isEmpty
        }
        let format = candidates.first {
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        } ?? candidates.first
        guard let format else { throw CaptureError.noSuitableFormat }

        // The largest Float32 depth format the video format supports.
        let depthFormat = format.supportedDepthDataFormats
            .filter { CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32 }
            .max { first, second in
                CMVideoFormatDescriptionGetDimensions(first.formatDescription).width
                    < CMVideoFormatDescriptionGetDimensions(second.formatDescription).width
            }
        guard let depthFormat else { throw CaptureError.noSuitableFormat }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            throw CaptureError.cannotConfigure(error.localizedDescription)
        }
        guard session.canAddInput(input) else { throw CaptureError.cannotConfigure("cannot add the camera input") }
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        // Unfiltered depth: holes stay NaN and are saved as 0 ("no reading") instead of being smoothed over.
        depthOutput.isFilteringEnabled = false
        depthOutput.alwaysDiscardsLateDepthData = true
        guard session.canAddOutput(videoOutput), session.canAddOutput(depthOutput) else {
            throw CaptureError.cannotConfigure("cannot add the video and depth outputs")
        }
        session.addOutput(videoOutput)
        session.addOutput(depthOutput)

        do {
            try device.lockForConfiguration()
        } catch {
            throw CaptureError.cannotConfigure(error.localizedDescription)
        }
        device.activeFormat = format
        device.activeDepthDataFormat = depthFormat
        device.videoZoomFactor = 1.0  // no digital zoom, ever
        device.unlockForConfiguration()

        // Each frame carries its own intrinsic matrix. Buffers are left in sensor orientation:
        // no rotation or mirroring is ever set on these connections.
        if let connection = videoOutput.connection(with: .video), connection.isCameraIntrinsicMatrixDeliverySupported {
            connection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }

        let synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer.setDelegate(self, queue: dataQueue)
        self.synchronizer = synchronizer
        self.device = device

        let depthSize = CMVideoFormatDescriptionGetDimensions(depthFormat.formatDescription)
        // Geometric distortion correction is left at the device default for this format and only
        // recorded, so capture and inference always agree.
        let result = CaptureConfiguration(
            videoWidth: Int(Self.videoWidth), videoHeight: Int(Self.videoHeight),
            depthWidth: Int(depthSize.width), depthHeight: Int(depthSize.height),
            geometricDistortionCorrectionEnabled:
                device.isGeometricDistortionCorrectionSupported && device.isGeometricDistortionCorrectionEnabled,
            videoZoomFactor: Double(device.videoZoomFactor))
        self.configuration = result
        return result
    }

    // MARK: Frames

    func latestFrame() -> CapturedFrame? {
        lock.lock()
        defer { lock.unlock() }
        return storedFrame
    }

    func currentLensPosition() -> Double? {
        guard let device else { return nil }
        return Double(device.lensPosition)
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        guard let device,
              let videoData = synchronizedDataCollection.synchronizedData(for: videoOutput)
                as? AVCaptureSynchronizedSampleBufferData,
              let depthData = synchronizedDataCollection.synchronizedData(for: depthOutput)
                as? AVCaptureSynchronizedDepthData,
              !videoData.sampleBufferWasDropped, !depthData.depthDataWasDropped,
              let image = CMSampleBufferGetImageBuffer(videoData.sampleBuffer)
        else { return }

        var depth = depthData.depthData
        if depth.depthDataType != kCVPixelFormatType_DepthFloat32 {
            depth = depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        }
        let depthMap = depth.depthDataMap
        let imageWidth = Double(CVPixelBufferGetWidth(image)), imageHeight = Double(CVPixelBufferGetHeight(image))
        let depthWidth = Double(CVPixelBufferGetWidth(depthMap)), depthHeight = Double(CVPixelBufferGetHeight(depthMap))

        // Calibration that travels with the depth data, valid at its own reference dimensions.
        var calibrated: (intrinsics: Intrinsics, width: Double, height: Double)?
        if let calibration = depth.cameraCalibrationData {
            let reference = calibration.intrinsicMatrixReferenceDimensions
            if reference.width > 0, reference.height > 0 {
                calibrated = (Self.intrinsics(from: calibration.intrinsicMatrix),
                              Double(reference.width), Double(reference.height))
            }
        }
        // The matrix attached to this very video frame, already in the frame's own pixels.
        var perFrame: Intrinsics?
        if let attachment = CMGetAttachment(videoData.sampleBuffer,
                                            key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
                                            attachmentModeOut: nil) as? Data,
           attachment.count >= MemoryLayout<matrix_float3x3>.size {
            let matrix = attachment.withUnsafeBytes { $0.loadUnaligned(as: matrix_float3x3.self) }
            perFrame = Self.intrinsics(from: matrix)
        }

        let intrinsicsImage: Intrinsics
        let source: (intrinsics: Intrinsics, width: Double, height: Double)
        if let calibrated {
            source = calibrated
            intrinsicsImage = perFrame ?? calibrated.intrinsics.scaled(
                fromWidth: calibrated.width, fromHeight: calibrated.height, toWidth: imageWidth, toHeight: imageHeight)
        } else if let perFrame {
            source = (perFrame, imageWidth, imageHeight)
            intrinsicsImage = perFrame
        } else {
            return  // no intrinsics at all: the frame is useless for geometry
        }
        let intrinsicsDepth = source.intrinsics.scaled(
            fromWidth: source.width, fromHeight: source.height, toWidth: depthWidth, toHeight: depthHeight)

        let frame = CapturedFrame(
            image: image, depth: depthMap,
            intrinsicsImage: intrinsicsImage, intrinsicsDepth: intrinsicsDepth,
            intrinsicsReferenceWidth: Int(source.width.rounded()), intrinsicsReferenceHeight: Int(source.height.rounded()),
            lensPosition: Double(device.lensPosition), isAdjustingFocus: device.isAdjustingFocus,
            iso: Double(device.iso), exposureDurationSeconds: CMTimeGetSeconds(device.exposureDuration),
            presentationTime: CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(videoData.sampleBuffer)),
            date: Date())
        lock.lock()
        storedFrame = frame
        lock.unlock()
    }

    /// fx, fy, cx, cy from a column-major 3×3 intrinsic matrix.
    private static func intrinsics(from matrix: matrix_float3x3) -> Intrinsics {
        Intrinsics(fx: Double(matrix.columns.0.x), fy: Double(matrix.columns.1.y),
                   cx: Double(matrix.columns.2.x), cy: Double(matrix.columns.2.y))
    }

    // MARK: Focus

    func focus(atDevicePoint point: CGPoint) async -> FocusResult {
        guard let device = await beginFocus(at: point) else {
            return FocusResult(lensPosition: currentLensPosition() ?? 0, settled: false)
        }
        // The lens may take a moment to start moving; if it never does, it was already in place.
        let started = Date()
        var sawAdjusting = false
        while Date().timeIntervalSince(started) < Self.focusTimeoutSeconds {
            try? await Task.sleep(nanoseconds: 50_000_000)
            if device.isAdjustingFocus {
                sawAdjusting = true
            } else if sawAdjusting || Date().timeIntervalSince(started) > 0.4 {
                try? await Task.sleep(nanoseconds: 100_000_000)  // let the lens come to rest
                return FocusResult(lensPosition: Double(device.lensPosition), settled: !device.isAdjustingFocus)
            }
        }
        return FocusResult(lensPosition: Double(device.lensPosition), settled: false)
    }

    /// Start a one-shot autofocus (and exposure metering) at the point. Returns nil if focus
    /// at a point is not possible.
    private func beginFocus(at point: CGPoint) async -> AVCaptureDevice? {
        await withCheckedContinuation { (continuation: CheckedContinuation<AVCaptureDevice?, Never>) in
            sessionQueue.async {
                guard let device = self.device, device.isFocusModeSupported(.autoFocus),
                      (try? device.lockForConfiguration()) != nil
                else {
                    continuation.resume(returning: nil)
                    return
                }
                if device.isFocusPointOfInterestSupported {
                    device.focusPointOfInterest = point
                }
                device.focusMode = .autoFocus
                if device.isExposurePointOfInterestSupported {
                    device.exposurePointOfInterest = point
                }
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
                device.unlockForConfiguration()
                continuation.resume(returning: device)
            }
        }
    }
}
