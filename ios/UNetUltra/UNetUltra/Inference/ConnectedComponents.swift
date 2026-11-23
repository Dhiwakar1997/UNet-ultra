import Foundation

/// One connected blob of a single class. `pixels` are row-major indices (`row * width + column`).
struct MaskComponent: Equatable {
    var classId: UInt8
    var pixels: [Int]

    var area: Int { pixels.count }
}

/// Connected-component labelling by flood fill, 4-connected like `scipy.ndimage.label`. Foundation-only.
enum ConnectedComponents {
    /// Blobs of equal non-zero class in a class map, largest first. Blobs under `minPixels` are dropped.
    static func components(classMap: [UInt8], width: Int, height: Int, minPixels: Int = 30) -> [MaskComponent] {
        guard width > 0, height > 0, classMap.count >= width * height else { return [] }
        var visited = [Bool](repeating: false, count: width * height)
        var found: [MaskComponent] = []
        var stack: [Int] = []
        for start in 0..<(width * height) where classMap[start] != 0 && !visited[start] {
            let classId = classMap[start]
            var pixels: [Int] = []
            visited[start] = true
            stack.append(start)
            while let index = stack.popLast() {
                pixels.append(index)
                let column = index % width
                if column > 0 { visit(index - 1, classId, classMap, &visited, &stack) }
                if column < width - 1 { visit(index + 1, classId, classMap, &visited, &stack) }
                if index >= width { visit(index - width, classId, classMap, &visited, &stack) }
                if index < width * (height - 1) { visit(index + width, classId, classMap, &visited, &stack) }
            }
            if pixels.count >= minPixels {
                pixels.sort()
                found.append(MaskComponent(classId: classId, pixels: pixels))
            }
        }
        return found.sorted { $0.area > $1.area }
    }

    private static func visit(_ index: Int, _ classId: UInt8, _ classMap: [UInt8],
                              _ visited: inout [Bool], _ stack: inout [Int]) {
        if !visited[index], classMap[index] == classId {
            visited[index] = true
            stack.append(index)
        }
    }

    /// The component under the tap, or the largest one if the tap hit none. `components` must be
    /// sorted largest first, as `components(classMap:...)` returns them.
    static func select(_ components: [MaskComponent], tapIndex: Int?) -> (component: MaskComponent, containsTap: Bool)? {
        if let tapIndex, let hit = components.first(where: { $0.pixels.contains(tapIndex) }) {
            return (hit, true)
        }
        guard let largest = components.first else { return nil }
        return (largest, false)
    }
}
