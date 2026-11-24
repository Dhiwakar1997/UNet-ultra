import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO

extension ComputeUnitsOption {
    var coreML: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        }
    }
}

/// `self_test/meta.json`: the meta values PyTorch was given for the self-test image.
struct SelfTestMeta: Codable, Equatable {
    var metaFeatures: [String]?
    var values: [Double]?

    enum CodingKeys: String, CodingKey {
        case metaFeatures = "meta_features"
        case values
    }
}

/// Runs a bundle's self-test on this device: the stored input goes through the compiled model
/// exactly as a camera frame would after cropping, and the output is compared with what PyTorch
/// produced on the laptop. A failure points at a pre-processing or conversion mismatch.
enum ModelSelfTest {
    static let inputName = "input.jpg"
    static let metaName = "meta.json"
    static let maskName = "expected_mask.png"
    static let depthName = "expected_depth.npy"

    /// - `folder`: the model's library folder, holding `model.mlmodelc` and `self_test/`.
    static func run(folder: URL, card: ModelCard, computeUnits: ComputeUnitsOption) -> SelfTestResult {
        let tests = folder.appendingPathComponent(ModelContract.selfTestFolder, isDirectory: true)
        let manager = FileManager.default
        let inputURL = tests.appendingPathComponent(inputName)
        let maskURL = tests.appendingPathComponent(maskName)
        guard manager.fileExists(atPath: inputURL.path), manager.fileExists(atPath: maskURL.path) else {
            return .failure("the bundle has no self_test/\(inputName) and self_test/\(maskName).")
        }
        guard let input = pixelBuffer(fromImageAt: inputURL) else {
            return .failure("self_test/\(inputName) could not be decoded.")
        }
        let width = CVPixelBufferGetWidth(input), height = CVPixelBufferGetHeight(input)
        guard width == card.input.width, height == card.input.height else {
            return .failure("self_test/\(inputName) is \(width)×\(height), but the model input is "
                + "\(card.input.width)×\(card.input.height).")
        }

        var meta: [Float] = []
        let wanted = card.metaFeatures ?? []
        if !wanted.isEmpty {
            guard let data = try? Data(contentsOf: tests.appendingPathComponent(metaName)),
                  let stored = try? JSONDecoder().decode(SelfTestMeta.self, from: data),
                  let values = stored.values, values.count == wanted.count
            else {
                return .failure("self_test/\(metaName) does not hold \(wanted.count) meta value(s).")
            }
            meta = values.map { Float($0) }
        }

        guard let expectedMask = grayBytes(fromImageAt: maskURL) else {
            return .failure("self_test/\(maskName) could not be decoded.")
        }
        guard expectedMask.width == width, expectedMask.height == height else {
            return .failure("self_test/\(maskName) is \(expectedMask.width)×\(expectedMask.height), not \(width)×\(height).")
        }

        let output: ModelOutput
        do {
            let model = try CoreMLSegmentationModel(
                compiledModelURL: folder.appendingPathComponent(ModelStoreLayout.compiledName, isDirectory: true),
                name: card.name, computeUnits: computeUnits.coreML, descriptor: card.descriptor())
            output = try model.predict(image: input, meta: meta)
        } catch {
            return .failure("the model could not be run (\(error.localizedDescription)).")
        }
        guard output.width == width, output.height == height else {
            return .failure("the model's output is \(output.width)×\(output.height), not \(width)×\(height).")
        }

        var expectedDepth: [Float]?
        let depthURL = tests.appendingPathComponent(depthName)
        if output.depth != nil, manager.fileExists(atPath: depthURL.path) {
            do {
                let array = try NpyReader.floats(from: Data(contentsOf: depthURL))
                guard array.values.count == width * height else {
                    return .failure("self_test/\(depthName) has shape \(array.shape), not \(height)×\(width).")
                }
                expectedDepth = array.values
            } catch {
                return .failure("self_test/\(depthName): \(error.localizedDescription)")
            }
        }
        return SelfTestComparison.compare(
            predictedMask: output.classMap, expectedMask: expectedMask.values,
            predictedDepth: expectedDepth == nil ? nil : output.depth, expectedDepth: expectedDepth)
    }

    /// Decode an image file into a 32BGRA pixel buffer at its own size: no resize, no crop.
    static func pixelBuffer(fromImageAt url: URL) -> CVPixelBuffer? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return nil }
        let width = image.width, height = image.height
        var created: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [String: Any](),
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                  attributes as CFDictionary, &created) == kCVReturnSuccess,
              let buffer = created
        else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: colorSpace, bitmapInfo: bitmapInfo)
        else { return nil }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    /// Decode an 8-bit grayscale image into row-major bytes. A plain 8-bit gray PNG is read
    /// as stored, so class ids survive exactly; anything else is drawn into a gray bitmap.
    static func grayBytes(fromImageAt url: URL) -> (values: [UInt8], width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return nil }
        let width = image.width, height = image.height
        guard width > 0, height > 0 else { return nil }
        var values = [UInt8](repeating: 0, count: width * height)
        if image.bitsPerComponent == 8, image.bitsPerPixel == 8, let provided = image.dataProvider?.data {
            let raw = provided as Data
            let bytesPerRow = image.bytesPerRow
            guard raw.count >= (height - 1) * bytesPerRow + width else { return nil }
            raw.withUnsafeBytes { bytes in
                for row in 0..<height {
                    for column in 0..<width {
                        values[row * width + column] = bytes[row * bytesPerRow + column]
                    }
                }
            }
            return (values, width, height)
        }
        let drawn: Bool = values.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(
                data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return drawn ? (values, width, height) : nil
    }
}
