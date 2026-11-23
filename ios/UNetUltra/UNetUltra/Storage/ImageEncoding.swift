import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Encodes captured buffers into the files of a sample. Nothing here rotates or resizes:
/// files are written exactly as the sensor delivered the frame.
enum ImageEncoding {
    private static let context = CIContext()

    /// JPEG (quality 0.95, sRGB) of a BGRA frame, in sensor orientation.
    static func jpegData(from pixelBuffer: CVPixelBuffer, quality: Double = 0.95) -> Data? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let qualityKey = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        return context.jpegRepresentation(of: image, colorSpace: colorSpace, options: [qualityKey: quality])
    }

    /// The frame converted to sRGB through the same Core Image path as `jpegData`, as a new BGRA
    /// buffer. Inference uses this so the model sees the colours the training JPEGs had.
    static func srgbPixelBuffer(from pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer)
        var created: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                  attributes as CFDictionary, &created) == kCVReturnSuccess,
              let output = created, let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        else { return nil }
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        context.render(image, to: output, bounds: image.extent, colorSpace: colorSpace)
        return output
    }

    /// Depth in metres -> millimetres, row-major. Invalid readings become 0.
    static func depthMillimetres(from frame: CapturedFrame) -> [UInt16] {
        frame.depthValues().map(DepthMath.millimetres(fromMetres:))
    }

    /// 16-bit grayscale PNG from row-major values (for `depth_mm.png`).
    ///
    /// The samples are handed to Core Graphics in the host's little-endian order and flagged
    /// as such, so ImageIO writes the big-endian samples PNG requires.
    static func png16Data(values: [UInt16], width: Int, height: Int) -> Data? {
        guard width > 0, height > 0, values.count == width * height else { return nil }
        let bytes = values.map { $0.littleEndian }.withUnsafeBufferPointer { Data(buffer: $0) }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
        guard let provider = CGDataProvider(data: bytes as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16,
                                  bytesPerRow: width * 2, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: bitmapInfo, provider: provider, decode: nil,
                                  shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return output as Data
    }

    /// Read a 16-bit grayscale PNG back into row-major values. Used by the tests and for debugging.
    static func decodePNG16(_ data: Data) -> (values: [UInt16], width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.bitsPerComponent == 16, image.bitsPerPixel == 16,
              let provided = image.dataProvider?.data
        else { return nil }
        let raw = provided as Data
        let width = image.width, height = image.height, bytesPerRow = image.bytesPerRow
        guard raw.count >= (height - 1) * bytesPerRow + width * 2 else { return nil }
        // Core Graphics reports 16-bit samples as big-endian unless the image says little-endian.
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let littleEndian = order == .byteOrder16Little
        var values = [UInt16](repeating: 0, count: width * height)
        for row in 0..<height {
            for column in 0..<width {
                let index = row * bytesPerRow + column * 2
                let first = UInt16(raw[raw.startIndex + index]), second = UInt16(raw[raw.startIndex + index + 1])
                values[row * width + column] = littleEndian ? (second << 8 | first) : (first << 8 | second)
            }
        }
        return (values, width, height)
    }
}
