import SwiftUI

@main
struct UNetUltraApp: App {
    /// One capture service for the whole app, shared by the Capture and Inference tabs.
    @StateObject private var captureService = CaptureService()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(captureService)
        }
    }
}

struct RootView: View {
    var body: some View {
        TabView {
            CaptureView()
                .tabItem { Label("Capture", systemImage: "camera") }
            InferenceView()
                .tabItem { Label("Inference", systemImage: "viewfinder") }
        }
    }
}
