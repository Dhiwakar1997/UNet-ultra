import SwiftUI

@main
struct UNetUltraApp: App {
    /// One capture service for the whole app, shared by the Capture and Inference tabs.
    @StateObject private var captureService = CaptureService()
    @StateObject private var sessionStore = SessionStore()
    /// The model library, shared by the Inference and Models tabs.
    @StateObject private var modelStore = ModelStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(captureService)
                .environmentObject(sessionStore)
                .environmentObject(modelStore)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var captureService: CaptureService
    @EnvironmentObject private var modelStore: ModelStore
    @Environment(\.scenePhase) private var scenePhase
    @State private var selectedTab = RootTab.capture

    enum RootTab: Hashable {
        case capture, inference, models
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            CaptureView()
                .tabItem { Label("Capture", systemImage: "camera") }
                .tag(RootTab.capture)
            InferenceView()
                .tabItem { Label("Inference", systemImage: "viewfinder") }
                .tag(RootTab.inference)
            ModelLibraryView()
                .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
                .tag(RootTab.models)
        }
        .task { await modelStore.loadIfNeeded() }
        // A .unetmodel opened from AirDrop or Files ("Open in UNet-ultra") is imported into the library.
        .onOpenURL { url in
            selectedTab = .models
            Task { await modelStore.importBundle(from: url) }
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
