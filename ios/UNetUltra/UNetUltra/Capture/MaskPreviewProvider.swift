import CoreGraphics
import Foundation

/// Hook for a future on-device mask preview (for example Apple's Core ML SAM 2.1 models).
///
/// Nothing implements this yet: at capture time the phone makes no masks, and training masks
/// come from SAM 2 on the Mac. A provider would be handed the frame and tap right after focus
/// settles and return a mask to draw over the preview. It must never change what is saved.
protocol MaskPreviewProvider {
    /// Returns a mask at the frame's image size (non-zero = object), or nil if none could be made.
    func previewMask(for frame: CapturedFrame, tapImagePoint: CGPoint) async -> [UInt8]?
}
