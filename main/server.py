"""
GaussianFeels Production Server

Production-ready server with FastAPI, WebSocket support, monitoring, and deployment capabilities.
Integrates all GaussianFeels components for real-world applications.
"""

import asyncio
import logging
import signal
from misc.utils.logging_config import setup_logging as _setup_logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# High-performance PyTorch defaults
try:
    import torch
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 on Ampere+ for speed
    torch.backends.cudnn.benchmark = True   # Autotune best conv algos
    try:
        torch.set_float32_matmul_precision('high')  # Prefer TF32 matmuls where available
    except Exception:
        pass
except Exception:
    pass

try:
    import os
    # Optionally raise CUDA concurrent connections for overlap
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
except Exception:
    pass

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .config import GaussianFeelsConfig
from .trainer import GaussianTrainer
from .datasets import DatasetRegistry

# Optional modules (guarded to avoid hard import failures during basic UI usage)
try:
    from .distributed import DistributedTrainer, setup_distributed_training  # type: ignore
except Exception:
    DistributedTrainer = None  # type: ignore
    setup_distributed_training = None  # type: ignore

try:
    from .optimization import create_optimized_trainer  # type: ignore
except Exception:
    create_optimized_trainer = None  # type: ignore

try:
    from .sensors import SensorManager, setup_default_sensors  # type: ignore
except Exception:
    SensorManager = None  # type: ignore
    setup_default_sensors = None  # type: ignore
# Optional comprehensive web viewer (not required for basic UI)
try:
    from .web_viewer import WebViewerManager  # noqa: F401
except Exception:
    WebViewerManager = None  # type: ignore
# NOTE: EvaluationSuite imports experimental trainer types; import lazily in endpoint

# Metrics (if Prometheus available)
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter('gaussianfeels_requests_total', 'Total requests', ['method', 'endpoint'])
    REQUEST_DURATION = Histogram('gaussianfeels_request_duration_seconds', 'Request duration')
    ACTIVE_CONNECTIONS = Gauge('gaussianfeels_active_connections', 'Active WebSocket connections')
    TRAINING_LOSS = Gauge('gaussianfeels_training_loss', 'Current training loss')
    GPU_MEMORY = Gauge('gaussianfeels_gpu_memory_gb', 'GPU memory usage in GB')
    GAUSSIANS_COUNT = Gauge('gaussianfeels_gaussians_count', 'Number of active Gaussians')

# Global state
app_state = {
    'config': None,
    'trainer': None,
    'sensor_manager': None,
    'web_manager': None,
    'redis_client': None,
    'active_connections': set(),
    'training_active': False,
    'training_paused': False,
    'shutdown_requested': False
}

# Pydantic models for API
class TrainingRequest(BaseModel):
    dataset: str
    object: str
    log: str = "00"
    max_steps: int = 1000
    distributed: bool = False
    optimization_level: str = "balanced"

class TrainingStatus(BaseModel):
    active: bool
    paused: bool
    step: int
    loss: float
    gaussians: int
    memory_usage: float
    elapsed_time: float

class EvaluationRequest(BaseModel):
    dataset: str
    object: str
    method: str = "gaussianfeels"
    steps: int = 100

class FeedbackRequest(BaseModel):
    message: str
    meta: Optional[Dict[str, Any]] = None

class DiagnosticsResponse(BaseModel):
    gpu_count: int
    device: str
    trainer_step: int
    avg_map_time_ms: float
    recent_fps: float

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

# Create FastAPI app
app = FastAPI(
    title="GaussianFeels Server",
    description="Production server for multi-modal Gaussian splatting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic security headers (Phase 4 hardening)
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Relaxed CSP for dev; tighten for prod
    response.headers["Content-Security-Policy"] = "default-src 'self' https://unpkg.com 'unsafe-inline' 'unsafe-eval' data:"
    return response

async def startup_event():
    """Initialize server components"""
    try:
        print("🚀 Starting GaussianFeels Server...")
        
        # Load configuration
        config_path = Path("configs/production.yaml")
        if config_path.exists():
            app_state['config'] = GaussianFeelsConfig.from_file(config_path)
        else:
            app_state['config'] = GaussianFeelsConfig()
        
        # Initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                app_state['redis_client'] = redis.Redis(host='redis', port=6379, decode_responses=True)
                app_state['redis_client'].ping()
                print("✅ Connected to Redis")
            except (ConnectionError, ImportError, AttributeError) as e:
                print(f"⚠️ Redis connection failed: {e}")
                app_state['redis_client'] = None
        
        # Initialize sensor manager
        if hasattr(app_state['config'], 'sensors') and getattr(app_state['config'].sensors, 'enabled', False):
            if setup_default_sensors is None:
                raise RuntimeError("Sensors enabled in config but sensor module not available")
            app_state['sensor_manager'] = setup_default_sensors()
            print("✅ Sensor manager initialized")
        
        print("✅ GaussianFeels Server started successfully")
        
    except (RuntimeError, ImportError, ValueError, FileNotFoundError) as e:
        print(f"❌ Server startup failed: {e}")
        traceback.print_exc()
        sys.exit(1)

async def shutdown_event():
    """Clean up server components"""
    print("🛑 Shutting down GaussianFeels Server...")
    
    app_state['shutdown_requested'] = True
    
    # Stop training
    if app_state['training_active']:
        app_state['training_active'] = False
        print("⏹️ Training stopped")
    
    # Stop sensors
    if app_state['sensor_manager']:
        app_state['sensor_manager'].stop_all_sensors()
        print("⏹️ Sensors stopped")
    
    # Close connections
    for websocket in app_state['active_connections'].copy():
        try:
            await websocket.close()
        except (ConnectionClosedError, RuntimeError) as e:
            logger.debug(f"Failed to close websocket connection: {e}")
    
    print("✅ Server shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": {
            "config": app_state['config'] is not None,
            "redis": app_state['redis_client'] is not None,
            "sensors": app_state['sensor_manager'] is not None,
            "training": app_state['training_active']
        }
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if app_state['shutdown_requested']:
        raise HTTPException(status_code=503, detail="Server shutting down")
    
    return {"status": "ready", "timestamp": time.time()}

# Metrics endpoint
if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        # Update dynamic metrics
        if app_state['trainer']:
            metrics = app_state['trainer'].get_performance_metrics()
            TRAINING_LOSS.set(metrics.get('recent_map_loss', 0))
            GPU_MEMORY.set(metrics.get('memory_usage_mb', 0) / 1024)
            GAUSSIANS_COUNT.set(metrics.get('num_gaussians', 0))
        
        ACTIVE_CONNECTIONS.set(len(app_state['active_connections']))
        
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Training endpoints
@app.post("/api/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training session"""
    if app_state['training_active']:
        raise HTTPException(status_code=400, detail="Training already active")
    
    try:
        # Load dataset
        registry = DatasetRegistry(Path("data"))
        config = GaussianFeelsConfig(
            dataset=request.dataset,
            object=request.object,
            log=request.log,
            training={"max_steps": request.max_steps}
        )
        dataset = registry.load_dataset(config)
        
        # Create trainer - require specific trainer types
        if request.distributed:
            if setup_distributed_training is None:
                raise RuntimeError("Distributed training requested but module not available")
            trainer = setup_distributed_training(config, dataset)
        elif request.optimization_level and request.optimization_level > 0:
            if create_optimized_trainer is None:
                raise RuntimeError("Optimized trainer requested but module not available")
            trainer = create_optimized_trainer(config, dataset, request.optimization_level)
        else:
            trainer = GaussianTrainer(config, dataset)
        
        app_state['trainer'] = trainer
        app_state['training_active'] = True
        app_state['training_paused'] = False
        
        # Start training in background
        background_tasks.add_task(run_training_loop)
        
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(method='POST', endpoint='/api/training/start').inc()
        
        return {"status": "training_started", "config": request.dict()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training start failed: {e}")

@app.post("/api/training/stop")
async def stop_training():
    """Stop training session"""
    if not app_state['training_active']:
        raise HTTPException(status_code=400, detail="No training active")
    app_state['training_active'] = False
    app_state['training_paused'] = False
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(method='POST', endpoint='/api/training/stop').inc()
    return {"status": "training_stopped"}
@app.post("/api/training/pause")
async def pause_training():
    if not app_state['training_active']:
        raise HTTPException(status_code=400, detail="No training active")
    app_state['training_paused'] = True
    return {"status": "training_paused"}

@app.post("/api/training/resume")
async def resume_training():
    if not app_state['training_active']:
        raise HTTPException(status_code=400, detail="No training active")
    app_state['training_paused'] = False
    return {"status": "training_resumed"}

@app.post("/api/training/checkpoint")
async def save_checkpoint():
    trainer = app_state.get('trainer')
    if not trainer:
        raise HTTPException(status_code=400, detail="No trainer available")
    try:
        out_dir = Path.cwd() / 'checkpoints'
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'checkpoint_step_{trainer.step}.pth'
        trainer.save_checkpoint(path)
        return {"status": "checkpoint_saved", "path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checkpoint failed: {e}")
    
    # unreachable legacy code removed

@app.get("/api/training/status")
async def get_training_status() -> TrainingStatus:
    """Get current training status"""
    if not app_state['trainer']:
        return TrainingStatus(
            active=False, paused=False, step=0, loss=0.0, gaussians=0, 
            memory_usage=0.0, elapsed_time=0.0
        )
    
    metrics = app_state['trainer'].get_performance_metrics()
    
    return TrainingStatus(
        active=app_state['training_active'],
        paused=app_state['training_paused'],
        step=metrics.get('step', 0),
        loss=metrics.get('recent_map_loss', 0.0),
        gaussians=metrics.get('num_gaussians', 0),
        memory_usage=metrics.get('memory_usage_mb', 0.0),
        elapsed_time=sum(app_state['trainer'].timings.get('map_step', []))
    )

# Evaluation endpoints
@app.post("/api/evaluation/run")
async def run_evaluation(request: EvaluationRequest):
    """Run evaluation on dataset"""
    try:
        from .evaluation import EvaluationSuite  # lazy import to avoid heavy deps on startup
        registry = DatasetRegistry(Path("data"))
        evaluation_suite = EvaluationSuite(registry)
        
        config = GaussianFeelsConfig(
            dataset=request.dataset,
            object=request.object,
            log=getattr(request, 'log', '00'),
            training={"max_steps": request.steps}
        )
        
        metrics = evaluation_suite.evaluate_method(config, request.method)
        
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(method='POST', endpoint='/api/evaluation/run').inc()
        
        return {
            "status": "evaluation_complete",
            "metrics": {
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "training_time": metrics.training_time,
                "memory_usage": metrics.memory_usage,
                "gaussians": metrics.gaussian_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

# Dataset browsing endpoints
@app.get("/api/datasets")
async def list_datasets():
    registry = DatasetRegistry(Path("data"))
    return {"datasets": registry.list_datasets()}

@app.get("/api/datasets/{dataset}/objects")
async def list_objects(dataset: str):
    registry = DatasetRegistry(Path("data"))
    return {"objects": registry.list_objects(dataset)}

@app.get("/api/datasets/{dataset}/{object}/logs")
async def list_logs(dataset: str, object: str):
    registry = DatasetRegistry(Path("data"))
    return {"logs": registry.list_logs(dataset, object)}

# Sensor endpoints
@app.get("/api/sensors/status")
async def get_sensor_status():
    """Get status of all sensors"""
    if not app_state['sensor_manager']:
        return {"sensors": [], "message": "Sensor manager not initialized"}
    
    status = app_state['sensor_manager'].get_sensor_status()
    return {"sensors": status}

@app.post("/api/sensors/start")
async def start_sensors():
    """Start all sensors"""
    if not app_state['sensor_manager']:
        raise HTTPException(status_code=400, detail="Sensor manager not available")
    
    app_state['sensor_manager'].start_all_sensors()
    return {"status": "sensors_started"}

@app.post("/api/sensors/stop")
async def stop_sensors():
    """Stop all sensors"""
    if not app_state['sensor_manager']:
        raise HTTPException(status_code=400, detail="Sensor manager not available")
    
    app_state['sensor_manager'].stop_all_sensors()
    return {"status": "sensors_stopped"}

# Feedback endpoint (basic user testing artifact collection)
@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest):
    try:
        feedback_dir = Path.cwd() / 'logs'
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = feedback_dir / 'feedback.txt'
        line = f"{time.time()}\t{request.message}\t{(request.meta or {})}\n"
        with open(feedback_file, 'a') as f:
            f.write(line)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")

@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics():
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpu_count = 0
    trainer = app_state.get('trainer')
    if not trainer:
        return DiagnosticsResponse(gpu_count=gpu_count, device='cpu', trainer_step=0, avg_map_time_ms=0.0, recent_fps=0.0)
    avg_map = 0.0
    if trainer.timings.get('map_step'):
        recent = trainer.timings['map_step'][-50:]
        if recent:
            avg_map = sum(recent) / len(recent)
    metrics = trainer.get_performance_metrics()
    return DiagnosticsResponse(
        gpu_count=gpu_count,
        device=str(trainer.device),
        trainer_step=int(trainer.step),
        avg_map_time_ms=avg_map * 1000.0,
        recent_fps=float(metrics.get('recent_fps', 0.0))
    )

# Simple help page
@app.get("/help", include_in_schema=False)
async def help_page():
    html = """
    <!doctype html><html><head><title>GaussianFeels Help</title></head>
    <body style='font-family:Arial, sans-serif; padding:24px;'>
      <h1>GaussianFeels Help</h1>
      <p>Use the left sidebar to Start/Pause/Resume/Stop/Checkpoint. Toggle overlays and adjust render budget or enable auto-tuning.</p>
      <p>Right sidebar shows metrics, selection details, sensors, and a feedback box to share notes.</p>
      <h2>Shortcuts</h2>
      <ul>
        <li>Space: Play/Pause (coming soon)</li>
        <li>Arrow Left/Right: Frame nav (coming soon)</li>
        <li>G: Toggle Gaussians (always on)</li>
        <li>T: Toggle tactile overlay</li>
        <li>V: Toggle RGB overlay</li>
      </ul>
    </body></html>
    """
    return HTMLResponse(content=html)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state['active_connections'].add(websocket)
    
    try:
        while not app_state['shutdown_requested']:
            # Send periodic updates
            if app_state['trainer']:
                metrics = app_state['trainer'].get_performance_metrics()
                await websocket.send_json({
                    "type": "training_update",
                    "data": metrics
                })
            
            if app_state['sensor_manager']:
                sensor_status = app_state['sensor_manager'].get_sensor_status()
                await websocket.send_json({
                    "type": "sensor_update",
                    "data": sensor_status
                })
            
            await asyncio.sleep(1.0)  # Update every second
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        app_state['active_connections'].discard(websocket)

# Visualization data endpoints for WebUI
@app.get("/api/gaussians")
async def get_gaussians():
    """Return current Gaussian field for web UI"""
    trainer = app_state.get('trainer')
    if not trainer or not hasattr(trainer, 'gaussian_field'):
        return {"points": [], "colors": [], "scales": [], "opacities": [], "count": 0}
    try:
        gaussians = trainer.gaussian_field.get_gaussians()
        if not gaussians:
            return {"points": [], "colors": [], "scales": [], "opacities": [], "count": 0}
        points = [g["position"].cpu().numpy().tolist() for g in gaussians]
        colors = [g["color"].cpu().numpy().tolist() for g in gaussians]
        scales = [g.get("scale").cpu().numpy().tolist() if hasattr(g.get("scale"), 'cpu') else g.get("scale", [0,0,0]) for g in gaussians]
        opacities = [float(g.get("opacity").cpu().numpy()) if hasattr(g.get("opacity"), 'cpu') else float(g.get("opacity", 1.0)) for g in gaussians]
        return {"points": points, "colors": colors, "scales": scales, "opacities": opacities, "count": len(points)}
    except Exception as e:
        return {"points": [], "colors": [], "scales": [], "opacities": [], "count": 0, "error": str(e)}

@app.get("/api/losses")
async def get_losses():
    """Return recent loss history"""
    trainer = app_state.get('trainer')
    if not trainer:
        return {"pose_losses": [], "map_losses": [], "steps": []}
    try:
        map_losses = trainer.map_losses[-200:]
        steps = list(range(max(0, trainer.step - len(map_losses)), trainer.step))
        return {"pose_losses": trainer.pose_losses[-200:], "map_losses": map_losses, "steps": steps}
    except Exception as e:
        return {"pose_losses": [], "map_losses": [], "steps": [], "error": str(e)}

@app.get("/api/frame/{frame_id}")
async def get_frame(frame_id: int):
    """Return a specific frame's images and metadata in base64 where appropriate"""
    import base64
    from io import BytesIO
    from PIL import Image
    trainer = app_state.get('trainer')
    if not trainer or not hasattr(trainer, 'dataset'):
        raise HTTPException(status_code=404, detail="No dataset available")
    dataset = trainer.dataset
    if frame_id < 0 or frame_id >= len(dataset):
        raise HTTPException(status_code=404, detail="Frame out of range")
    frame = dataset[frame_id]
    def img_to_b64(img):
        try:
            if img.dtype != 'uint8':
                import numpy as np
                img = (img.astype(np.float32).clip(0, 1) * 255).astype('uint8')
            pil = Image.fromarray(img)
            buf = BytesIO()
            pil.save(buf, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None
    rgb_images = {}
    if frame.rgb_images:
        for name, img in frame.rgb_images.items():
            b64 = img_to_b64(img)
            if b64:
                rgb_images[name] = b64
    depth_images = {}
    if frame.depth_images:
        for name, img in frame.depth_images.items():
            b64 = img_to_b64(img if len(img.shape)==3 else (img*255.0/img.max() if img.max()>0 else img).astype('uint8'))
            if b64:
                depth_images[name] = b64
    tactile_images = {}
    if frame.tactile_images:
        for name, img in frame.tactile_images.items():
            b64 = img_to_b64(img)
            if b64:
                tactile_images[name] = b64
    return {
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "rgb_images": rgb_images,
        "depth_images": depth_images,
        "tactile_images": tactile_images,
        "metadata": frame.metadata or {}
    }

static_root = Path(__file__).resolve().parents[1] / "webui" / "dist"
if static_root.exists():
    # Serve built SPA at root so absolute asset URLs like /assets/... resolve
    app.mount("/", StaticFiles(directory=str(static_root), html=True), name="webui")

# Fail-fast if web UI not built - no fallback
@app.get("/", include_in_schema=False)
async def get_web_interface():
    """Serve built UI - fails if not available (no fallback)"""
    if not static_root.exists():
        raise HTTPException(status_code=503, detail="Web UI not built - run build process")
    index_path = static_root / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=503, detail="Web UI index.html missing")
    return FileResponse(index_path)

async def run_training_loop():
    """Background training loop"""
    trainer = app_state['trainer']
    if not trainer:
        return
    
    try:
        step = 0
        while app_state['training_active'] and step < 10000:
            if app_state['training_paused']:
                await asyncio.sleep(0.05)
                continue
            if step % 2 == 0:
                trainer.step_pose()
            else:
                trainer.step_map()
            
            step += 1
            
            # Brief pause to allow other operations
            await asyncio.sleep(0.01)
            
        app_state['training_active'] = False
        print(f"✅ Training completed after {step} steps")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        app_state['training_active'] = False

def setup_logging():
    """Setup production logging using centralized configuration"""
    try:
        log_dir = Path('/app/logs') if Path('/app/logs').exists() else Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'gaussianfeels.log'
        _setup_logging(level="INFO", log_file=str(log_file))
    except Exception as e:
        # Fallback to stdout-only logging
        _setup_logging(level="INFO")

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}, initiating shutdown...")
    app_state['shutdown_requested'] = True

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Run server (port configurable via env, default 8082 to avoid conflicts)
    import os
    port = int(os.environ.get("GFEELS_PORT", os.environ.get("PORT", "8082")))
    uvicorn.run(
        "gaussianfeels.server:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        loop="uvloop",
        log_level="info",
        access_log=True
    )


def main():
    """Main entry point for server CLI"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="GaussianFeels Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8082, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Override with environment variables
    host = os.environ.get("GFEELS_HOST", args.host)
    port = int(os.environ.get("GFEELS_PORT", os.environ.get("PORT", args.port)))
    
    print(f"🚀 Starting GaussianFeels server on {host}:{port}")
    
    try:
        import uvicorn
        uvicorn.run(
            "gaussianfeels.server:app",
            host=host,
            port=port,
            workers=args.workers,
            reload=args.dev,
            loop="uvloop" if not args.dev else "asyncio",
            log_level="debug" if args.dev else "info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())