import Foundation

/// CRC-32 (IEEE 802.3, polynomial 0xEDB88320), as used by ZIP.
enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { index -> UInt32 in
        var value = UInt32(index)
        for _ in 0..<8 {
            value = (value & 1) == 1 ? (0xEDB8_8320 ^ (value >> 1)) : (value >> 1)
        }
        return value
    }

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
        }
        return crc ^ 0xFFFF_FFFF
    }
}

enum ZipError: Error, Equatable {
    case cannotCreate(String)
    case tooLarge
    case tooManyEntries
    case badName(String)
}

/// Minimal ZIP writer: STORED (uncompressed) entries, no zip64, UTF-8 names.
///
/// Entries are written one at a time, so only one file is ever held in memory. JPEG and PNG
/// are already compressed, so storing them loses almost nothing.
final class ZipWriter {
    private struct Entry {
        let name: Data
        let crc: UInt32
        let size: UInt32
        let offset: UInt32
        let dosTime: UInt16
        let dosDate: UInt16
    }

    private let handle: FileHandle
    private var entries: [Entry] = []
    private var offset: UInt64 = 0
    private var finished = false

    /// Bit 11: file names are UTF-8.
    private static let flags: UInt16 = 0x0800
    private static let versionNeeded: UInt16 = 20

    init(url: URL) throws {
        let manager = FileManager.default
        if manager.fileExists(atPath: url.path) {
            try manager.removeItem(at: url)
        }
        guard manager.createFile(atPath: url.path, contents: nil) else {
            throw ZipError.cannotCreate(url.path)
        }
        handle = try FileHandle(forWritingTo: url)
    }

    /// Add one entry. `name` uses forward slashes and is relative to the zip root.
    func addEntry(name: String, data: Data, modified: Date = Date()) throws {
        guard !name.isEmpty, !name.hasPrefix("/"), !name.contains("\\") else { throw ZipError.badName(name) }
        let nameData = Data(name.utf8)
        guard nameData.count <= Int(UInt16.max) else { throw ZipError.badName(name) }
        guard entries.count < Int(UInt16.max) else { throw ZipError.tooManyEntries }
        let headerSize = UInt64(30 + nameData.count)
        guard data.count <= Int(UInt32.max), offset + headerSize + UInt64(data.count) < UInt64(UInt32.max) else {
            throw ZipError.tooLarge
        }
        let (dosTime, dosDate) = Self.dosTimestamp(modified)
        let entry = Entry(name: nameData, crc: CRC32.checksum(data), size: UInt32(data.count),
                          offset: UInt32(offset), dosTime: dosTime, dosDate: dosDate)

        var header = Data()
        header.appendLE(UInt32(0x0403_4B50))  // local file header signature
        header.appendLE(Self.versionNeeded)
        header.appendLE(Self.flags)
        header.appendLE(UInt16(0))  // method: stored
        header.appendLE(entry.dosTime)
        header.appendLE(entry.dosDate)
        header.appendLE(entry.crc)
        header.appendLE(entry.size)  // compressed size
        header.appendLE(entry.size)  // uncompressed size
        header.appendLE(UInt16(nameData.count))
        header.appendLE(UInt16(0))  // extra field length
        header.append(nameData)

        try handle.write(contentsOf: header)
        try handle.write(contentsOf: data)
        offset += UInt64(header.count) + UInt64(data.count)
        entries.append(entry)
    }

    /// Add a file from disk as one entry.
    func addFile(at url: URL, name: String) throws {
        let data = try Data(contentsOf: url)
        let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        try addEntry(name: name, data: data, modified: modified ?? Date())
    }

    /// Write the central directory and close the file. Must be called exactly once.
    func finish() throws {
        guard !finished else { return }
        finished = true
        let directoryOffset = offset
        var directory = Data()
        for entry in entries {
            directory.appendLE(UInt32(0x0201_4B50))  // central directory header signature
            directory.appendLE(Self.versionNeeded)  // version made by
            directory.appendLE(Self.versionNeeded)
            directory.appendLE(Self.flags)
            directory.appendLE(UInt16(0))  // method: stored
            directory.appendLE(entry.dosTime)
            directory.appendLE(entry.dosDate)
            directory.appendLE(entry.crc)
            directory.appendLE(entry.size)
            directory.appendLE(entry.size)
            directory.appendLE(UInt16(entry.name.count))
            directory.appendLE(UInt16(0))  // extra field length
            directory.appendLE(UInt16(0))  // comment length
            directory.appendLE(UInt16(0))  // disk number start
            directory.appendLE(UInt16(0))  // internal attributes
            directory.appendLE(UInt32(0))  // external attributes
            directory.appendLE(entry.offset)
            directory.append(entry.name)
        }
        guard directoryOffset + UInt64(directory.count) < UInt64(UInt32.max) else { throw ZipError.tooLarge }
        let directorySize = UInt32(directory.count)
        directory.appendLE(UInt32(0x0605_4B50))  // end of central directory signature
        directory.appendLE(UInt16(0))  // this disk
        directory.appendLE(UInt16(0))  // disk with the central directory
        directory.appendLE(UInt16(entries.count))
        directory.appendLE(UInt16(entries.count))
        directory.appendLE(directorySize)
        directory.appendLE(UInt32(directoryOffset))
        directory.appendLE(UInt16(0))  // comment length
        try handle.write(contentsOf: directory)
        try handle.close()
    }

    /// MS-DOS time and date words (local time, two-second resolution, years from 1980).
    static func dosTimestamp(_ date: Date) -> (time: UInt16, date: UInt16) {
        let parts = Calendar(identifier: .gregorian).dateComponents(
            [.year, .month, .day, .hour, .minute, .second], from: date)
        let year = min(max((parts.year ?? 1980) - 1980, 0), 127)
        let time = UInt16((parts.hour ?? 0) << 11 | (parts.minute ?? 0) << 5 | (parts.second ?? 0) / 2)
        let day = UInt16(year << 9 | (parts.month ?? 1) << 5 | (parts.day ?? 1))
        return (time, day)
    }
}

extension Data {
    /// Append a fixed-width integer in little-endian byte order.
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
