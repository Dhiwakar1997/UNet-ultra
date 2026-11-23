import AVFoundation
import SwiftUI
import UIKit

/// Converts between points in the preview view and normalized capture-device coordinates
/// ((0, 0) top-left to (1, 1) bottom-right of the sensor-orientation image), whatever the
/// preview's rotation and aspect fitting.
final class PreviewProxy {
    fileprivate weak var view: PreviewUIView?

    func layerPoint(fromDevicePoint point: CGPoint) -> CGPoint? {
        view?.previewLayer.layerPointConverted(fromCaptureDevicePoint: point)
    }

    func devicePoint(fromLayerPoint point: CGPoint) -> CGPoint? {
        view?.previewLayer.captureDevicePointConverted(fromLayerPoint: point)
    }
}

/// A UIView backed by an `AVCaptureVideoPreviewLayer`.
final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        // The layer class above guarantees this cast.
        layer as! AVCaptureVideoPreviewLayer
    }

    var onTap: ((CGPoint) -> Void)?

    @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
        let devicePoint = previewLayer.captureDevicePointConverted(fromLayerPoint: recognizer.location(in: self))
        onTap?(devicePoint)
    }
}

/// Live camera preview. Reports taps in normalized capture-device coordinates. Only the
/// preview is rotated to suit the portrait UI; captured buffers stay in sensor orientation.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let proxy: PreviewProxy
    let onTap: (CGPoint) -> Void

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        view.addGestureRecognizer(UITapGestureRecognizer(target: view, action: #selector(PreviewUIView.handleTap(_:))))
        view.onTap = onTap
        proxy.view = view
        return view
    }

    func updateUIView(_ view: PreviewUIView, context: Context) {
        view.onTap = onTap
        proxy.view = view
        // Show the landscape sensor image upright in the portrait UI.
        if let connection = view.previewLayer.connection, connection.isVideoRotationAngleSupported(90),
           connection.videoRotationAngle != 90 {
            connection.videoRotationAngle = 90
        }
    }
}
