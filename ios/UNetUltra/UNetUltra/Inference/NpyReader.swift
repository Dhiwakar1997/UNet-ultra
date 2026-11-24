import Foundation

enum NpyError: Error, LocalizedError, Equatable {
    case notNpy
    case unsupported(String)
    case truncated

    var errorDescription: String? {
        switch self {
        case .notNpy: return "The file is not a NumPy .npy array."
        case .unsupported(let what): return "The .npy array is \(what); expected little-endian float32 in C order."
        case .truncated: return "The .npy file is shorter than its header says."
        }
    }
}

/// Reads the one kind of `.npy` file a model bundle holds: little-endian float32, C order.
enum NpyReader {
    static func floats(from data: Data) throws -> (values: [Float], shape: [Int]) {
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]  // "\x93NUMPY"
        guard data.count >= 10, Array(data.prefix(6)) == magic else { throw NpyError.notNpy }
        let major = data[data.startIndex + 6]
        let headerLength: Int, headerStart: Int
        if major == 1 {
            headerLength = Int(data.le16(8))
            headerStart = 10
        } else if major == 2 || major == 3 {
            guard data.count >= 12 else { throw NpyError.truncated }
            headerLength = Int(data.le32(8))
            headerStart = 12
        } else {
            throw NpyError.unsupported("format version \(major)")
        }
        guard data.count >= headerStart + headerLength else { throw NpyError.truncated }
        let headerData = data.subdata(in: (data.startIndex + headerStart)..<(data.startIndex + headerStart + headerLength))
        guard let header = String(data: headerData, encoding: .utf8) ?? String(data: headerData, encoding: .isoLatin1) else {
            throw NpyError.notNpy
        }

        guard let descr = value(of: "descr", in: header) else { throw NpyError.notNpy }
        let type = descr.trimmingCharacters(in: CharacterSet(charactersIn: "'\" "))
        guard type == "<f4" else { throw NpyError.unsupported("of type \(type)") }
        if let order = value(of: "fortran_order", in: header), order.contains("True") {
            throw NpyError.unsupported("in Fortran order")
        }
        guard let shapeText = value(of: "shape", in: header) else { throw NpyError.notNpy }
        let shape = shapeText.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        let count = shape.reduce(1, *)
        let start = headerStart + headerLength
        guard data.count >= start + count * 4 else { throw NpyError.truncated }

        var values = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for index in 0..<count {
                let bits = raw.loadUnaligned(fromByteOffset: start + index * 4, as: UInt32.self)
                values[index] = Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
        return (values, shape)
    }

    /// The text after `'key':` in the header dict, up to the next top-level comma (a tuple's
    /// own commas are skipped) or the closing brace.
    private static func value(of key: String, in header: String) -> String? {
        guard let keyRange = header.range(of: "'\(key)'") ?? header.range(of: "\"\(key)\""),
              let colon = header[keyRange.upperBound...].firstIndex(of: ":")
        else { return nil }
        var depth = 0
        var text = ""
        for character in header[header.index(after: colon)...] {
            if character == "(" { depth += 1 }
            if character == ")" {
                depth -= 1
                text.append(character)
                if depth <= 0 { break }
                continue
            }
            if (character == "," || character == "}") && depth == 0 { break }
            text.append(character)
        }
        return text.trimmingCharacters(in: .whitespaces)
    }
}
