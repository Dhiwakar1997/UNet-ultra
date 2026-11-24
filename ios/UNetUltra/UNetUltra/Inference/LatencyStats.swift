import Foundation

/// How Core ML is allowed to run a model. Raw values are what the CSV log records.
enum ComputeUnitsOption: String, CaseIterable, Identifiable, Codable {
    case all = "all"
    case cpuAndNeuralEngine = "cpu_and_neural_engine"
    case cpuAndGPU = "cpu_and_gpu"
    case cpuOnly = "cpu_only"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: return "All (default)"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuOnly: return "CPU only"
        }
    }

    var shortTitle: String {
        switch self {
        case .all: return "All"
        case .cpuAndNeuralEngine: return "CPU+NE"
        case .cpuAndGPU: return "CPU+GPU"
        case .cpuOnly: return "CPU"
        }
    }
}

/// Rolling latency numbers for one model on one compute-units setting. Foundation-only.
struct LatencyStats: Equatable {
    static let capacity = 30

    private(set) var samples: [Double] = []

    var count: Int { samples.count }
    var last: Double? { samples.last }

    /// Add one measurement in milliseconds, keeping only the most recent `capacity`.
    mutating func add(_ milliseconds: Double) {
        guard milliseconds.isFinite else { return }
        samples.append(milliseconds)
        if samples.count > Self.capacity { samples.removeFirst(samples.count - Self.capacity) }
    }

    var median: Double? { samples.isEmpty ? nil : Self.quantile(sorted: samples.sorted(), 0.5) }
    var p95: Double? { samples.isEmpty ? nil : Self.quantile(sorted: samples.sorted(), 0.95) }

    /// Linear-interpolated quantile of an ascending array (the NumPy default). The array must not be empty.
    static func quantile(sorted values: [Double], _ q: Double) -> Double {
        guard values.count > 1 else { return values.first ?? .nan }
        let position = min(max(q, 0), 1) * Double(values.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(lower + 1, values.count - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - Double(lower))
    }

    var summary: String {
        guard let median, let p95 else { return "no runs yet" }
        return String(format: "median %.0f ms · p95 %.0f ms (n=%d)", median, p95, count)
    }
}
