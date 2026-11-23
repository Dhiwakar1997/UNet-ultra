import XCTest
@testable import UNetUltra

final class CaptureMathTests: XCTestCase {
    func testIntrinsicsScaleByPlainRatio() {
        let reference = Intrinsics(fx: 3000, fy: 3000, cx: 2016, cy: 1512)
        let image = reference.scaled(fromWidth: 4032, fromHeight: 3024, toWidth: 1920, toHeight: 1440)
        XCTAssertEqual(image.fx, 3000 * 1920.0 / 4032.0, accuracy: 1e-9)
        XCTAssertEqual(image.fy, 3000 * 1440.0 / 3024.0, accuracy: 1e-9)
        XCTAssertEqual(image.cx, 960, accuracy: 1e-9)
        XCTAssertEqual(image.cy, 720, accuracy: 1e-9)
    }

    func testScalingToDepthDirectlyOrViaImageAgrees() {
        let reference = Intrinsics(fx: 3050, fy: 3048, cx: 2020.5, cy: 1508.25)
        let direct = reference.scaled(fromWidth: 4032, fromHeight: 3024, toWidth: 320, toHeight: 240)
        let viaImage = reference
            .scaled(fromWidth: 4032, fromHeight: 3024, toWidth: 1920, toHeight: 1440)
            .scaled(fromWidth: 1920, fromHeight: 1440, toWidth: 320, toHeight: 240)
        XCTAssertEqual(direct.fx, viaImage.fx, accuracy: 1e-9)
        XCTAssertEqual(direct.cx, viaImage.cx, accuracy: 1e-9)
        XCTAssertEqual(direct.cy, viaImage.cy, accuracy: 1e-9)
    }

    func testMillimetresTreatsBadReadingsAsNoReading() {
        XCTAssertEqual(DepthMath.millimetres(fromMetres: 0.3004), 300)
        XCTAssertEqual(DepthMath.millimetres(fromMetres: 0.3006), 301)
        XCTAssertEqual(DepthMath.millimetres(fromMetres: .nan), 0)
        XCTAssertEqual(DepthMath.millimetres(fromMetres: .infinity), 0)
        XCTAssertEqual(DepthMath.millimetres(fromMetres: 0), 0)
        XCTAssertEqual(DepthMath.millimetres(fromMetres: -1), 0)
        XCTAssertEqual(DepthMath.millimetres(fromMetres: 70), 0)  // beyond 16-bit millimetres
    }

    func testMedianDepthIgnoresInvalidValuesAndClampsAtEdges() {
        // 4×3 map: 0.5 m everywhere, with a hole and one outlier near the top-left corner.
        var values = [Float](repeating: 0.5, count: 12)
        values[0] = .nan
        values[1] = 3.0
        XCTAssertEqual(DepthMath.medianDepth(values: values, width: 4, height: 3, x: 0, y: 0, window: 3) ?? -1,
                       0.5, accuracy: 1e-6)
        XCTAssertEqual(DepthMath.medianDepth(values: values, width: 4, height: 3, x: 99, y: 99, window: 3) ?? -1,
                       0.5, accuracy: 1e-6)
        XCTAssertNil(DepthMath.medianDepth(values: [Float](repeating: .nan, count: 12), width: 4, height: 3, x: 1, y: 1))
        XCTAssertNil(DepthMath.medianDepth(values: [], width: 4, height: 3, x: 1, y: 1))
    }

    func testDepthPixelMapping() {
        let centre = DepthMath.depthPixel(forImageX: 960, y: 720, imageWidth: 1920, imageHeight: 1440,
                                          depthWidth: 320, depthHeight: 240)
        XCTAssertEqual(centre.x, 160)
        XCTAssertEqual(centre.y, 120)
        let corner = DepthMath.depthPixel(forImageX: 1919.9, y: 1439.9, imageWidth: 1920, imageHeight: 1440,
                                          depthWidth: 320, depthHeight: 240)
        XCTAssertEqual(corner.x, 319)
        XCTAssertEqual(corner.y, 239)
    }

    func testDepthPNGRoundTripKeepsEveryValue() throws {
        // Values chosen so that a byte-order mistake cannot go unnoticed (300 mm = 0x012C).
        let width = 5, height = 3
        var values = [UInt16](repeating: 0, count: width * height)
        for index in values.indices { values[index] = UInt16(300 + index * 257) }
        values[4] = 0
        values[7] = 65534
        let png = try XCTUnwrap(ImageEncoding.png16Data(values: values, width: width, height: height))
        // PNG signature, then an IHDR saying 16-bit grayscale: bit depth 16, colour type 0.
        XCTAssertEqual(Array(png.prefix(8)), [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        XCTAssertEqual(png[png.startIndex + 24], 16)
        XCTAssertEqual(png[png.startIndex + 25], 0)
        let decoded = try XCTUnwrap(ImageEncoding.decodePNG16(png))
        XCTAssertEqual(decoded.width, width)
        XCTAssertEqual(decoded.height, height)
        XCTAssertEqual(decoded.values, values)
    }
}
