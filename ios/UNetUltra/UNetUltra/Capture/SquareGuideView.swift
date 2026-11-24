import SwiftUI

/// Shows on the live preview exactly which part of the frame the model sees: the centre
/// square of the sensor image. Everything outside it is dimmed.
///
/// The square is drawn through the preview layer's own point conversion, so it stays right
/// whatever the preview's rotation and aspect fitting.
struct SquareGuideView: View {
    let proxy: PreviewProxy
    let imageWidth: Int
    let imageHeight: Int

    var body: some View {
        // The preview layer's geometry settles a moment after the view appears; a slow
        // timeline makes sure the guide is drawn once it has.
        TimelineView(.periodic(from: .now, by: 0.5)) { timeline in
            Canvas { context, size in
                // Reading the date makes each tick a real redraw; without it the square can be
                // left where the preview used to be after the layout changes.
                _ = timeline.date
                guard let corners = Self.corners(proxy: proxy, imageWidth: imageWidth, imageHeight: imageHeight)
                else { return }
                var square = Path()
                square.move(to: corners[0])
                for corner in corners.dropFirst() { square.addLine(to: corner) }
                square.closeSubpath()
                var dimmed = Path(CGRect(origin: .zero, size: size))
                dimmed.addPath(square)
                context.fill(dimmed, with: .color(.black.opacity(0.5)), style: FillStyle(eoFill: true))
                context.stroke(square, with: .color(.white.opacity(0.9)), lineWidth: 1.5)
            }
        }
        .allowsHitTesting(false)
    }

    /// The square's four corners in the preview's coordinates, in drawing order.
    static func corners(proxy: PreviewProxy, imageWidth: Int, imageHeight: Int) -> [CGPoint]? {
        guard imageWidth > 0, imageHeight > 0 else { return nil }
        let rect = CenterSquareCrop(sourceWidth: imageWidth, sourceHeight: imageHeight,
                                    outputWidth: 1, outputHeight: 1).normalizedRect
        let devicePoints = [
            CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY),
            CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.maxY),
        ]
        let converted = devicePoints.compactMap { proxy.layerPoint(fromDevicePoint: $0) }
        return converted.count == 4 ? converted : nil
    }
}
