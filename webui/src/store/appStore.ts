import { create } from 'zustand'

type Metrics = {
  step?: number
  recent_map_loss?: number
  recent_fps?: number
  gpu_utilization?: number
  num_gaussians?: number
  memory_usage_mb?: number
}

type AppState = {
  ws?: WebSocket
  connected: boolean
  connectionError: boolean
  metrics: Metrics
  fpsHistory: number[]
  gaussians: { points: Float32Array; colors: Float32Array; scales?: Float32Array; opacities?: Float32Array } | null
  showOverlays: { rgb: boolean; tactile: boolean }
  maxPoints: number
  currentFrame: number
  selectedIndex: number | null
  frameData?: { rgb_images?: Record<string,string>; depth_images?: Record<string,string>; tactile_images?: Record<string,string> }
  autoTunePerformance: boolean
  targetFps: number
  connect: () => void
  disconnect: () => void
  fetchGaussians: () => Promise<void>
  fetchFrame: () => Promise<void>
  toggleOverlay: (key: 'rgb' | 'tactile') => void
  setMaxPoints: (n: number) => void
  setCurrentFrame: (f: number) => void
  setSelectedIndex: (i: number | null) => void
  toggleAutoTune: () => void
}

export const useAppStore = create<AppState>((set, get) => ({
  ws: undefined,
  connected: false,
  connectionError: false,
  metrics: {},
  fpsHistory: [],
  gaussians: null,
  showOverlays: { rgb: true, tactile: false },
  maxPoints: 20000,
  currentFrame: 0,
  selectedIndex: null,
  autoTunePerformance: true,
  targetFps: 60,
  connect: () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)
      ws.onopen = () => set({ connected: true, connectionError: false })
      ws.onclose = () => {
        set({ connected: false, ws: undefined, connectionError: true })
        // Auto-reconnect after a short delay
        setTimeout(() => {
          if (!get().connected) get().connect()
        }, 2000)
      }
      ws.onerror = () => set({ connected: false, connectionError: true })
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === 'training_update') {
            const m = msg.data as any
            set((s) => {
              const fps = Math.max(0, Number(m.recent_fps ?? 0))
              const fpsHistory = [...s.fpsHistory, fps].slice(-300)
              // Auto-tune render budget
              let nextMax = s.maxPoints
              if (s.autoTunePerformance) {
                const target = s.targetFps
                if (fps < target * 0.8) {
                  nextMax = Math.max(2000, Math.floor(s.maxPoints * 0.85))
                } else if (fps > target * 0.95) {
                  nextMax = Math.min(100000, Math.floor(s.maxPoints * 1.1))
                }
              }
              return {
                metrics: {
                  step: m.step,
                  recent_map_loss: m.recent_map_loss,
                  recent_fps: m.recent_fps,
                  gpu_utilization: m.gpu_utilization,
                  num_gaussians: m.num_gaussians,
                  memory_usage_mb: m.memory_usage_mb
                },
                fpsHistory,
                maxPoints: nextMax
              }
            })
          }
        } catch {}
      }
      set({ ws })
    } catch {}
  },
  disconnect: () => {
    const ws = get().ws
    try { ws?.close() } catch {}
    set({ ws: undefined, connected: false })
  },
  fetchGaussians: async () => {
    try {
      const res = await fetch('/api/gaussians')
      const data = await res.json()
      const points = new Float32Array((data.points ?? []).flat())
      const colors = new Float32Array((data.colors ?? []).flat())
      const scales = new Float32Array((data.scales ?? []).flat())
      const opacities = new Float32Array((data.opacities ?? []).flat())
      set({ gaussians: { points, colors, scales, opacities } })
    } catch {}
  },
  fetchFrame: async () => {
    try {
      const idx = get().currentFrame
      const res = await fetch(`/api/frame/${idx}`)
      const data = await res.json()
      set({ frameData: { rgb_images: data.rgb_images, depth_images: data.depth_images, tactile_images: data.tactile_images } })
    } catch {}
  },
  toggleOverlay: (key) => set(s => ({ showOverlays: { ...s.showOverlays, [key]: !s.showOverlays[key] } })),
  setMaxPoints: (n) => set({ maxPoints: Math.max(1000, Math.min(100000, n)) }),
  setCurrentFrame: (f) => set({ currentFrame: Math.max(0, f|0) }),
  setSelectedIndex: (i) => set({ selectedIndex: i }),
  toggleAutoTune: () => set(s => ({ autoTunePerformance: !s.autoTunePerformance }))
}))


