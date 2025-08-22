"""
GaussianFeels Web Viewer

Real-time web-based visualization for Gaussian splatting training.
Includes FastAPI backend, WebSocket streaming, and interactive 3D visualization.
"""

import json
import asyncio
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

try:
    from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    WEB_DEPS_AVAILABLE = True
except ImportError:
    WEB_DEPS_AVAILABLE = False

from .config import GaussianFeelsConfig
from .trainer import GaussianSplattingTrainer

class WebViewerManager:
    """Manages web-based viewing and real-time updates"""
    
    def __init__(self, config: GaussianFeelsConfig, trainer: GaussianSplattingTrainer):
        self.config = config
        self.trainer = trainer
        self.app = None
        self.websocket_connections = []
        self.server_process = None
        # STRICT: Web viewer config must have explicit port and host
        if not hasattr(config.viewer, 'port'):
            raise ValueError("Config.viewer missing required 'port' attribute")
        if not hasattr(config.viewer, 'host'):
            raise ValueError("Config.viewer missing required 'host' attribute")
        
        self.port = config.viewer.port
        self.host = config.viewer.host
        
        if not WEB_DEPS_AVAILABLE:
            raise ImportError("Web viewer dependencies not available. Install with: pip install fastapi uvicorn websockets")
        
        self._setup_app()
        
    def _setup_app(self):
        """Setup FastAPI application"""
        self.app = FastAPI(title="GaussianFeels Web Viewer", version="1.0.0")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Serve static files
        static_dir = Path(__file__).parent / "web_static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_viewer():
            """Serve the main viewer page"""
            return self._get_viewer_html()
        
        @self.app.get("/api/config")
        async def get_config():
            """Get current configuration"""
            return {
                "dataset": self.config.dataset,
                "mode": self.config.mode,
                "modality": self.config.modality,
                "object": getattr(self.config, 'object', 'unknown'),
                "max_gaussians": self.config.training.max_gaussians,
                "device": self.config.device,
                "viewer": self.config.viewer.__dict__ if self.config.viewer else {}
            }
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current training status"""
            metrics = self.trainer.get_performance_metrics()
            return {
                "step": self.trainer.step,
                "running": self.trainer.is_training,
                "gaussians": self.trainer.num_gaussians,
                "metrics": metrics
            }
        
        @self.app.get("/api/gaussians")
        async def get_gaussians():
            """Get current Gaussian field data"""
            gaussians = self.trainer.gaussian_field.get_gaussians()
            
            if not gaussians:
                return {"points": [], "colors": [], "scales": [], "opacities": []}
            
            # Convert to JSON-serializable format
            points = [g["position"].cpu().numpy().tolist() for g in gaussians]
            colors = [g["color"].cpu().numpy().tolist() for g in gaussians]
            scales = [g["scale"].cpu().numpy().tolist() for g in gaussians]
            opacities = [float(g["opacity"].cpu().numpy()) for g in gaussians]
            
            return {
                "points": points,
                "colors": colors,
                "scales": scales,
                "opacities": opacities,
                "count": len(gaussians)
            }
        
        @self.app.get("/api/frame/{frame_id}")
        async def get_frame(frame_id: int):
            """Get specific frame data"""
            if frame_id >= len(self.trainer.dataset):
                raise HTTPException(status_code=404, detail="Frame not found")
            
            frame = self.trainer.dataset[frame_id]
            
            # Convert images to base64
            frame_data = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "rgb_images": {},
                "tactile_images": {},
                "metadata": frame.metadata
            }
            
            if frame.rgb_images:
                for cam_name, img in frame.rgb_images.items():
                    frame_data["rgb_images"][cam_name] = self._image_to_base64(img)
            
            if frame.tactile_images:
                for sensor_name, img in frame.tactile_images.items():
                    frame_data["tactile_images"][sensor_name] = self._image_to_base64(img)
            
            return frame_data
        
        @self.app.get("/api/losses")
        async def get_losses():
            """Get loss history"""
            return {
                "pose_losses": self.trainer.pose_losses[-100:],  # Last 100 steps
                "map_losses": self.trainer.map_losses[-100:],
                "steps": list(range(max(0, len(self.trainer.map_losses) - 100), len(self.trainer.map_losses)))
            }
        
        @self.app.post("/api/control/{action}")
        async def control_training(action: str):
            """Control training (start/stop/reset)"""
            if action == "pause":
                self.trainer.pause_training()
            elif action == "resume":
                self.trainer.resume_training()
            elif action == "reset":
                self.trainer.reset()
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
            
            return {"status": "success", "action": action}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await self._send_websocket_update(websocket)
                    await asyncio.sleep(0.5)  # 2 FPS updates
                    
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    def _image_to_base64(self, img: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(img.shape) == 3:
            pil_img = Image.fromarray(img, mode='RGB')
        else:
            pil_img = Image.fromarray(img, mode='L')
        
        # Convert to base64
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    async def _send_websocket_update(self, websocket: WebSocket):
        """Send real-time update via WebSocket"""
        try:
            metrics = self.trainer.get_performance_metrics()
            
            update = {
                "type": "update",
                "step": self.trainer.step,
                "gaussians": self.trainer.num_gaussians,
                "metrics": metrics,
                "timestamp": metrics.get("timestamp", 0)
            }
            
            await websocket.send_json(update)
            
        except Exception as e:
            print(f"Failed to send WebSocket update: {e}")
    
    async def broadcast_update(self, update_data: Dict):
        """Broadcast update to all connected clients"""
        if not self.websocket_connections:
            return
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(update_data)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)
    
    def _get_viewer_html(self) -> str:
        """Generate the main viewer HTML page"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GaussianFeels Web Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; background: #000; }
        #container { width: 100vw; height: 100vh; display: flex; }
        #viewer { flex: 1; position: relative; }
        #sidebar { width: 300px; background: #222; color: #fff; padding: 20px; overflow-y: auto; }
        #controls { margin-bottom: 20px; }
        #metrics { margin-bottom: 20px; }
        #lossChart { width: 100%; height: 200px; }
        .button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; margin: 5px; }
        .button:hover { background: #45a049; }
        .metric { margin: 5px 0; }
        #status { padding: 10px; background: #333; margin-bottom: 10px; border-radius: 5px; }
        #imageGrid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
        .imageContainer { text-align: center; }
        .imageContainer img { max-width: 100%; height: auto; border: 1px solid #444; }
        .imageContainer h4 { margin: 5px 0; font-size: 12px; }
    </style>
</head>
<body>
    <div id="container">
        <div id="viewer"></div>
        <div id="sidebar">
            <h2>GaussianFeels Viewer</h2>
            
            <div id="status">
                <div id="statusText">Connecting...</div>
            </div>
            
            <div id="controls">
                <button class="button" onclick="pauseTraining()">Pause</button>
                <button class="button" onclick="resumeTraining()">Resume</button>
                <button class="button" onclick="resetTraining()">Reset</button>
            </div>
            
            <div id="metrics">
                <h3>Metrics</h3>
                <div class="metric">Step: <span id="step">0</span></div>
                <div class="metric">Gaussians: <span id="gaussians">0</span></div>
                <div class="metric">FPS: <span id="fps">0.0</span></div>
                <div class="metric">Map Loss: <span id="mapLoss">0.0</span></div>
                <div class="metric">GPU: <span id="gpu">0.0%</span></div>
            </div>
            
            <div id="chartContainer">
                <h3>Loss History</h3>
                <canvas id="lossChart"></canvas>
            </div>
            
            <div id="imageGrid">
                <div class="imageContainer">
                    <h4>RGB Camera</h4>
                    <img id="rgbImage" src="" alt="RGB" style="display: none;">
                </div>
                <div class="imageContainer">
                    <h4>Tactile Sensor</h4>
                    <img id="tactileImage" src="" alt="Tactile" style="display: none;">
                </div>
            </div>
        </div>
    </div>

    <script>
        // Three.js setup
        let scene, camera, renderer, pointCloud;
        let gui, controls = {};
        let ws;
        let lossChart;
        let currentFrame = 0;
        
        init();
        animate();
        setupWebSocket();
        setupChart();
        
        function init() {
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 5);
            
            // Create renderer
            const viewerElement = document.getElementById('viewer');
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(viewerElement.clientWidth, viewerElement.clientHeight);
            viewerElement.appendChild(renderer.domElement);
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Add coordinate axes
            const axesHelper = new THREE.AxesHelper(1);
            scene.add(axesHelper);
            
            // Window resize handler
            window.addEventListener('resize', onWindowResize, false);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            // Rotate camera around scene
            const time = Date.now() * 0.0005;
            camera.position.x = Math.cos(time) * 5;
            camera.position.z = Math.sin(time) * 5;
            camera.lookAt(scene.position);
            
            renderer.render(scene, camera);
        }
        
        function onWindowResize() {
            const viewerElement = document.getElementById('viewer');
            camera.aspect = viewerElement.clientWidth / viewerElement.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewerElement.clientWidth, viewerElement.clientHeight);
        }
        
        function setupWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function() {
                document.getElementById('statusText').textContent = 'Connected';
                updateGaussians();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateMetrics(data);
            };
            
            ws.onclose = function() {
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(setupWebSocket, 2000); // Reconnect after 2 seconds
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                document.getElementById('statusText').textContent = 'Error';
            };
        }
        
        function setupChart() {
            const ctx = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Map Loss',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        borderWidth: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: { color: '#444' },
                            ticks: { color: '#fff' }
                        },
                        x: {
                            grid: { color: '#444' },
                            ticks: { color: '#fff' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#fff' } }
                    }
                }
            });
        }
        
        function updateMetrics(data) {
            if (data.step !== undefined) {
                document.getElementById('step').textContent = data.step;
            }
            if (data.gaussians !== undefined) {
                document.getElementById('gaussians').textContent = data.gaussians.toLocaleString();
            }
            if (data.metrics) {
                const metrics = data.metrics;
                if (metrics.recent_fps !== undefined) {
                    document.getElementById('fps').textContent = metrics.recent_fps.toFixed(1);
                }
                if (metrics.recent_map_loss !== undefined) {
                    document.getElementById('mapLoss').textContent = metrics.recent_map_loss.toFixed(6);
                }
                if (metrics.gpu_utilization !== undefined) {
                    document.getElementById('gpu').textContent = metrics.gpu_utilization.toFixed(1) + '%';
                }
            }
            
            // Update loss chart periodically
            if (data.step % 10 === 0) {
                updateLossChart();
            }
            
            // Update Gaussians periodically
            if (data.step % 5 === 0) {
                updateGaussians();
            }
            
            // Update frame images
            updateFrameImages();
        }
        
        async function updateGaussians() {
            try {
                const response = await fetch('/api/gaussians');
                const data = await response.json();
                
                // Remove existing point cloud
                if (pointCloud) {
                    scene.remove(pointCloud);
                }
                
                if (data.points && data.points.length > 0) {
                    // Create geometry
                    const geometry = new THREE.BufferGeometry();
                    
                    // Convert data to Float32Array
                    const positions = new Float32Array(data.points.flat());
                    const colors = new Float32Array(data.colors.flat());
                    
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    
                    // Create material
                    const material = new THREE.PointsMaterial({
                        size: 0.02,
                        vertexColors: true,
                        sizeAttenuation: true
                    });
                    
                    // Create point cloud
                    pointCloud = new THREE.Points(geometry, material);
                    scene.add(pointCloud);
                }
            } catch (error) {
                console.error('Failed to update Gaussians:', error);
            }
        }
        
        async function updateLossChart() {
            try {
                const response = await fetch('/api/losses');
                const data = await response.json();
                
                lossChart.data.labels = data.steps;
                lossChart.data.datasets[0].data = data.map_losses;
                lossChart.update('none');
            } catch (error) {
                console.error('Failed to update loss chart:', error);
            }
        }
        
        async function updateFrameImages() {
            try {
                const response = await fetch(`/api/frame/${currentFrame}`);
                const data = await response.json();
                
                // Update RGB image
                if (data.rgb_images && Object.keys(data.rgb_images).length > 0) {
                    const firstRgb = Object.values(data.rgb_images)[0];
                    const rgbImg = document.getElementById('rgbImage');
                    rgbImg.src = firstRgb;
                    rgbImg.style.display = 'block';
                }
                
                // Update tactile image
                if (data.tactile_images && Object.keys(data.tactile_images).length > 0) {
                    const firstTactile = Object.values(data.tactile_images)[0];
                    const tactileImg = document.getElementById('tactileImage');
                    tactileImg.src = firstTactile;
                    tactileImg.style.display = 'block';
                }
                
                // Cycle through frames
                currentFrame = (currentFrame + 1) % 10; // Cycle through first 10 frames
                
            } catch (error) {
                console.error('Failed to update frame images:', error);
            }
        }
        
        async function pauseTraining() {
            try {
                await fetch('/api/control/pause', { method: 'POST' });
            } catch (error) {
                console.error('Failed to pause training:', error);
            }
        }
        
        async function resumeTraining() {
            try {
                await fetch('/api/control/resume', { method: 'POST' });
            } catch (error) {
                console.error('Failed to resume training:', error);
            }
        }
        
        async function resetTraining() {
            try {
                await fetch('/api/control/reset', { method: 'POST' });
            } catch (error) {
                console.error('Failed to reset training:', error);
            }
        }
    </script>
</body>
</html>
        """
    
    async def start_server(self):
        """Start the web server"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        print(f"🌐 Starting web viewer at http://{self.host}:{self.port}")
        await server.serve()
    
    def start_background(self):
        """Start web server in background"""
        import threading
        
        def run_server():
            asyncio.run(self.start_server())
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        print(f"🌐 Web viewer started at http://{self.host}:{self.port}")
        return thread