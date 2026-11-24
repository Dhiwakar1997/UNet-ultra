import Foundation

enum ZipReadError: Error, LocalizedError, Equatable {
    case notAZip
    case unsupported(String)
    case compressedEntry(String)
    case unsafePath(String)
    case corrupt(String)
    case checksumMismatch(String)

    var errorDescription: String? {
        switch self {
        case .notAZip: return "The file is not a zip archive."
        case .unsupported(let what): return "The archive uses \(what), which this app cannot read."
        case .compressedEntry(let name):
            return "\"\(name)\" is compressed. Model bundles must be stored without compression; re-export the model."
        case .unsafePath(let name): return "The archive holds an unsafe path (\"\(name)\") and was refused."
        case .corrupt(let what): return "The archive is damaged: \(what)."
        case .checksumMismatch(let name): return "\"\(name)\" failed its checksum; the file was damaged in transfer."
        }
    }
}

/// Minimal ZIP reader, the counterpart of `ZipWriter`: STORED entries only, no zip64, no
/// encryption. Entries are streamed to disk in pieces, so a large model never sits in memory.
struct ZipReader {
    struct Entry: Equatable {
        let name: String
        let size: UInt64
        let crc: UInt32
        let method: UInt16
        let localHeaderOffset: UInt64

        var isDirectory: Bool { name.hasSuffix("/") }
    }

    let url: URL
    let entries: [Entry]

    private static let chunkSize = 1 << 20

    init(url: URL) throws {
        self.url = url
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = try handle.seekToEnd()
        guard fileSize >= 22 else { throw ZipReadError.notAZip }

        // The end-of-central-directory record sits at the end, after an optional comment.
        let tailLength = min(fileSize, UInt64(22 + 65535))
        try handle.seek(toOffset: fileSize - tailLength)
        let tail = try handle.read(upToCount: Int(tailLength)) ?? Data()
        guard let recordStart = Self.findEndRecord(in: tail) else { throw ZipReadError.notAZip }
        let count = Int(tail.le16(recordStart + 10))
        let directorySize = UInt64(tail.le32(recordStart + 12))
        let directoryOffset = UInt64(tail.le32(recordStart + 16))
        guard count != 0xFFFF, directoryOffset != 0xFFFF_FFFF else { throw ZipReadError.unsupported("zip64") }
        guard directoryOffset + directorySize <= fileSize else { throw ZipReadError.corrupt("central directory out of range") }

        try handle.seek(toOffset: directoryOffset)
        let directory = try handle.read(upToCount: Int(directorySize)) ?? Data()
        var parsed: [Entry] = []
        var position = 0
        for _ in 0..<count {
            guard position + 46 <= directory.count, directory.le32(position) == 0x0201_4B50 else {
                throw ZipReadError.corrupt("bad central directory entry")
            }
            let flags = directory.le16(position + 8)
            let method = directory.le16(position + 10)
            let crc = directory.le32(position + 16)
            let compressedSize = directory.le32(position + 20)
            let size = directory.le32(position + 24)
            let nameLength = Int(directory.le16(position + 28))
            let extraLength = Int(directory.le16(position + 30))
            let commentLength = Int(directory.le16(position + 32))
            let offset = directory.le32(position + 42)
            guard position + 46 + nameLength <= directory.count else { throw ZipReadError.corrupt("truncated entry name") }
            let nameData = directory.subdata(in: (directory.startIndex + position + 46)..<(directory.startIndex + position + 46 + nameLength))
            guard let name = String(data: nameData, encoding: .utf8) else { throw ZipReadError.corrupt("entry name is not UTF-8") }
            if flags & 0x0001 != 0 { throw ZipReadError.unsupported("encryption") }
            if size == 0xFFFF_FFFF || compressedSize == 0xFFFF_FFFF || offset == 0xFFFF_FFFF {
                throw ZipReadError.unsupported("zip64")
            }
            try Self.checkSafe(name)
            let isDirectory = name.hasSuffix("/")
            if !isDirectory && (method != 0 || compressedSize != size) { throw ZipReadError.compressedEntry(name) }
            parsed.append(Entry(name: name, size: UInt64(size), crc: crc, method: method, localHeaderOffset: UInt64(offset)))
            position += 46 + nameLength + extraLength + commentLength
        }
        entries = parsed
    }

    /// Names must stay inside the destination: no absolute paths, no "..", no backslashes.
    static func checkSafe(_ name: String) throws {
        guard !name.isEmpty, !name.hasPrefix("/"), !name.contains("\\"), !name.contains("\0") else {
            throw ZipReadError.unsafePath(name)
        }
        for part in name.split(separator: "/", omittingEmptySubsequences: false) where part == ".." {
            throw ZipReadError.unsafePath(name)
        }
    }

    private static func findEndRecord(in tail: Data) -> Int? {
        guard tail.count >= 22 else { return nil }
        var index = tail.count - 22
        while index >= 0 {
            if tail.le32(index) == 0x0605_4B50 { return index }
            index -= 1
        }
        return nil
    }

    /// Extract every entry under `directory`, creating folders as needed and checking each
    /// file's CRC-32. `progress` is called with (bytes written, total bytes).
    func extractAll(to directory: URL, progress: ((UInt64, UInt64) -> Void)? = nil) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let total = entries.reduce(UInt64(0)) { $0 + $1.size }
        var written: UInt64 = 0
        for entry in entries {
            let target = directory.appendingPathComponent(entry.name, isDirectory: entry.isDirectory)
            if entry.isDirectory {
                try manager.createDirectory(at: target, withIntermediateDirectories: true)
                continue
            }
            try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)

            // The local header repeats the name and may carry its own extra field.
            try handle.seek(toOffset: entry.localHeaderOffset)
            let header = try handle.read(upToCount: 30) ?? Data()
            guard header.count == 30, header.le32(0) == 0x0403_4B50 else {
                throw ZipReadError.corrupt("bad local header for \(entry.name)")
            }
            let dataOffset = entry.localHeaderOffset + 30 + UInt64(header.le16(26)) + UInt64(header.le16(28))
            try handle.seek(toOffset: dataOffset)

            guard manager.createFile(atPath: target.path, contents: nil) else {
                throw ZipReadError.corrupt("cannot create \(entry.name)")
            }
            let output = try FileHandle(forWritingTo: target)
            defer { try? output.close() }
            var remaining = entry.size
            var crc = CRC32.start
            while remaining > 0 {
                let wanted = Int(min(remaining, UInt64(Self.chunkSize)))
                let chunk = try handle.read(upToCount: wanted) ?? Data()
                guard !chunk.isEmpty else { throw ZipReadError.corrupt("\(entry.name) is cut short") }
                crc = CRC32.update(crc, with: chunk)
                try output.write(contentsOf: chunk)
                remaining -= UInt64(chunk.count)
                written += UInt64(chunk.count)
                progress?(written, total)
            }
            guard CRC32.finish(crc) == entry.crc else { throw ZipReadError.checksumMismatch(entry.name) }
        }
    }
}

extension Data {
    /// Little-endian integers at a byte offset from the start of this data.
    func le16(_ offset: Int) -> UInt16 {
        let base = startIndex + offset
        return UInt16(self[base]) | UInt16(self[base + 1]) << 8
    }

    func le32(_ offset: Int) -> UInt32 {
        let base = startIndex + offset
        return UInt32(self[base]) | UInt32(self[base + 1]) << 8 | UInt32(self[base + 2]) << 16 | UInt32(self[base + 3]) << 24
    }
}
