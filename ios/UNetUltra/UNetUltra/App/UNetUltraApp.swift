import SwiftUI

@main
struct UNetUltraApp: App {
    /// One capture service for the whole app, shared by the Capture and Inference tabs.
    @StateObject private var captureService = CaptureService()
    @StateObject private var sessionStore = SessionStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(captureService)
                .environmentObject(sessionStore)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var captureService: CaptureService
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        TabView {
            CaptureView()
                .tabItem { Label("Capture", systemImage: "camera") }
            InferenceView()
                .tabItem { Label("Inference", systemImage: "viewfinder") }
        }
        // The camera runs only while the app is in front; both tabs share the one session.
        .onChange(of: scenePhase) { _, phase in
            Task {
                switch phase {
                case .active: await captureService.start()
                case .background: await captureService.stop()
                default: break
                }
            }
        }
    }
}
