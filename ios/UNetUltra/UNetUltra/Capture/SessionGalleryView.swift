import ImageIO
import SwiftUI
import UIKit

/// Thumbnails of the current session's samples with their tap markers, and delete.
struct SessionGalleryView: View {
    @EnvironmentObject private var store: SessionStore
    @Environment(\.dismiss) private var dismiss

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 8)]

    var body: some View {
        NavigationStack {
            ScrollView {
                Text("\(store.sampleCount) samples, \(store.backgroundOnlyCount) no-object")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(store.samples, id: \.sampleId) { meta in
                        SampleThumbnail(meta: meta, imageURL: store.imageURL(meta.sampleId))
                            .contextMenu {
                                Button("Delete", systemImage: "trash", role: .destructive) {
                                    store.deleteSample(meta.sampleId)
                                }
                            }
                    }
                }
                .padding(8)
                Text("Touch and hold a sample to delete it. Images are stored as the sensor delivers them (landscape).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            }
            .navigationTitle("Gallery")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
    }
}

/// One sample: its image (sensor orientation, 4:3) with a dot at each tap point.
struct SampleThumbnail: View {
    let meta: SampleMeta
    let imageURL: URL?

    @State private var thumbnail: UIImage?

    var body: some View {
        VStack(spacing: 2) {
            ZStack {
                Color.black
                if let thumbnail {
                    Image(uiImage: thumbnail).resizable().scaledToFit()
                }
                GeometryReader { geometry in
                    ForEach(Array(meta.objects.enumerated()), id: \.offset) { index, object in
                        if object.tapPointImage.count == 2, meta.imageSize.count == 2 {
                            Text("\(index + 1)")
                                .font(.caption2.bold())
                                .foregroundStyle(.black)
                                .padding(3)
                                .background(.yellow, in: Circle())
                                .position(
                                    x: geometry.size.width * object.tapPointImage[0] / Double(meta.imageSize[0]),
                                    y: geometry.size.height * object.tapPointImage[1] / Double(meta.imageSize[1]))
                        }
                    }
                }
            }
            .aspectRatio(4.0 / 3.0, contentMode: .fit)
            Text(caption).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .task(id: meta.sampleId) {
            guard let imageURL else { return }
            thumbnail = await Task.detached { Self.loadThumbnail(imageURL, maxPixel: 480) }.value
        }
    }

    private var caption: String {
        if meta.backgroundOnly { return "\(meta.sampleId) · no object" }
        let distance = meta.objects.first?.lidarDistanceAtTapM.map { String(format: " · %.0f cm", $0 * 100) } ?? ""
        return "\(meta.sampleId) · \(meta.objects.count) obj\(distance)"
    }

    /// Decode a small version of the image without loading it at full size.
    nonisolated static func loadThumbnail(_ url: URL, maxPixel: Int) -> UIImage? {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
            kCGImageSourceCreateThumbnailWithTransform: false,
        ]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        else { return nil }
        return UIImage(cgImage: image)
    }
}
