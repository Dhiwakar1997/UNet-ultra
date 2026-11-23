import AVFoundation
import Foundation

/// Owns the AVFoundation capture session (LiDAR depth camera, synchronized video + depth).
///
/// Phase 0 only reports whether the LiDAR camera exists; the session itself is built in phase 1.
/// AVFoundation is used instead of ARKit so that the lens is free to focus and `lensPosition`
/// carries information.
@MainActor
final class CaptureService: ObservableObject {
    enum Availability: Equatable {
        case available
        case unavailable(reason: String)
    }

    @Published private(set) var availability: Availability

    init() {
        availability = Self.detectAvailability()
    }

    var isLiDARAvailable: Bool { availability == .available }

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
}
