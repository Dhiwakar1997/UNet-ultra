import CoreGraphics
import SwiftUI

/// Builds the tinted image of a predicted mask, at the model's resolution.
enum MaskImage {
    /// Premultiplied RGBA image: the component's pixels in a translucent tint, the rest clear.
    static func make(pixels: [Int], width: Int, height: Int,
                     red: UInt8 = 255, green: UInt8 = 59, blue: UInt8 = 48, alpha: UInt8 = 128) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let scale = Double(alpha) / 255
        let r = UInt8(Double(red) * scale), g = UInt8(Double(green) * scale), b = UInt8(Double(blue) * scale)
        for index in pixels where index >= 0 && index < width * height {
            bytes[index * 4] = r
            bytes[index * 4 + 1] = g
            bytes[index * 4 + 2] = b
            bytes[index * 4 + 3] = alpha
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

/// Draws a model-resolution mask over the live preview, on the square the model saw.
///
/// The mask is in sensor orientation while the preview is rotated for the portrait UI, so the
/// image is drawn through an affine transform built from three corners of the crop square
/// converted by the preview layer itself. Rotation and aspect fitting therefore come out right
/// without the mask ever being rotated in memory.
struct MaskOverlayView: View {
    let image: CGImage
    let crop: CenterSquareCrop
    let proxy: PreviewProxy

    var body: some View {
        Canvas { context, _ in
            let rect = crop.normalizedRect
            guard let origin = proxy.layerPoint(fromDevicePoint: CGPoint(x: rect.minX, y: rect.minY)),
                  let right = proxy.layerPoint(fromDevicePoint: CGPoint(x: rect.maxX, y: rect.minY)),
                  let down = proxy.layerPoint(fromDevicePoint: CGPoint(x: rect.minX, y: rect.maxY))
            else { return }
            let width = CGFloat(image.width), height = CGFloat(image.height)
            // Maps the image's (0,0), (width,0) and (0,height) onto origin, right and down.
            context.concatenate(CGAffineTransform(
                a: (right.x - origin.x) / width, b: (right.y - origin.y) / width,
                c: (down.x - origin.x) / height, d: (down.y - origin.y) / height,
                tx: origin.x, ty: origin.y))
            context.draw(Image(decorative: image, scale: 1), in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        .allowsHitTesting(false)
    }
}
