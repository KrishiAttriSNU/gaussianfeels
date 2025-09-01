"""
GaussianFeels Visualization System

Comprehensive visualization for Gaussian splatting training with multiple viewer backends.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, List

import numpy as np

# Import tkinter - if not available, disable tkinter viewer
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️ Tkinter not available - Tkinter viewer disabled, using Open3D/Web only")

if TYPE_CHECKING:
    from .trainer import GaussianSplattingTrainer

from .config import GaussianFeelsConfig, ViewerConfig

# Check dependencies
try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

class BaseViewer(ABC):
    """Abstract base class for all viewers"""
    
    def __init__(self, config: GaussianFeelsConfig, trainer: 'GaussianSplattingTrainer'):
        self.config = config
        self.trainer = trainer
        self.running = False
        
    def start(self):
        """Start the viewer"""
        print("Base viewer starting...")
    
    def update(self):
        """Update the viewer with latest data"""
        print("Base viewer updating...")
    
    def stop(self):
        """Stop the viewer"""
        print("Base viewer stopping...")

class HeadlessViewer(BaseViewer):
    """Headless viewer (no visualization)"""
    
    def start(self):
        """Start headless viewer"""
        self.running = True
        print("✅ Headless viewer started (no visualization)")
    
    def update(self):
        """Update headless viewer"""
        # Headless viewer has no visualization to update
        # This is intentionally a no-op
        return
    
    def stop(self):
        """Stop headless viewer"""
        self.running = False

class Open3DViewer(BaseViewer):
    """Open3D-based interactive real-time viewer with training controls"""
    
    def __init__(self, config: GaussianFeelsConfig, trainer: 'GaussianSplattingTrainer'):
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D not available. Install with: pip install open3d")
        
        super().__init__(config, trainer)
        self.window = None
        self.scene = None
        self.material = None
        
        # Geometry objects
        self.gaussian_cloud = None
        self.tactile_cloud = None
        self.camera_frustums = []
        self.reconstructed_mesh = None
        
        # GUI controls
        self.training_running = True
        self.step_counter = 0
        self.update_interval = 5  # Update every N steps
        
        # Performance tracking
        self.fps_history = []
        self.loss_history = []
        
        # Threading for GUI updates
        self.gui_thread = None
        
    def start(self):
        """Start Open3D interactive GUI viewer"""
        if self.running:
            return
        
        try:
            print("⠋ Starting interactive Open3D viewer...")
            
            # Start GUI in separate thread
            self.gui_thread = threading.Thread(target=self._run_gui, daemon=True)
            self.gui_thread.start()
            
            # Wait for GUI to initialize
            time.sleep(1.0)
            
            self.running = True
            print("✅ Interactive Open3D viewer started")
            
        except Exception as e:
            print(f"❌ Failed to start Open3D viewer: {e}")
            raise
    
    def _run_gui(self):
        """Run the GUI in its own thread"""
        app = gui.Application.instance
        app.initialize()
        
        # Create main window
        self._create_main_window()
        
        # Run the application
        app.run()
    
    def _create_main_window(self):
        """Create the main GUI window with controls"""
        # Get screen dimensions
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        
        # Create window
        self.window = gui.Application.instance.create_window(
            f"GaussianFeels Interactive Viewer - {self.config.object}", 
            int(w * 0.8), int(h * 0.8)
        )
        
        # Setup layouts
        self._setup_layouts()
        
        # Initialize 3D scene
        self._setup_3d_scene()
        
        # Setup controls panel
        self._setup_control_panel()
        
        # Setup status panel
        self._setup_status_panel()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Start update timer
        self.window.set_on_tick_event(self._on_tick)
    
    def _setup_layouts(self):
        """Setup the window layout"""
        # Main horizontal layout
        em = self.window.theme.font_size
        main_layout = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
        
        # Left panel for controls (300px width)
        self.controls_panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self.controls_panel.preferred_width = 300
        
        # Right panel for 3D view
        self.scene_panel = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        
        main_layout.add_child(self.controls_panel)
        main_layout.add_child(self.scene_panel)
        
        self.window.add_child(main_layout)
    
    def _setup_3d_scene(self):
        """Setup the 3D visualization scene"""
        # Create 3D scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene = rendering.Open3DScene(self.scene_widget.get_render())
        
        # Setup camera
        bounds = o3d.geometry.AxisAlignedBoundingBox([-2, -2, -2], [2, 2, 2])
        self.scene.camera.look_at([0, 0, 0], [0, 0, -2], [0, 1, 0])
        self.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Setup lighting
        self.scene.set_lighting(self.scene.LightingProfile.BRIGHT_DAY, [0, -1, -1])
        
        # Create materials
        self.material = rendering.MaterialRecord()
        self.material.albedo_img = None
        self.material.shader = "defaultUnlit"
        self.material.point_size = 3.0
        
        # Add scene to panel
        self.scene_panel.add_child(self.scene_widget)
    
    def _setup_control_panel(self):
        """Setup the control panel with training parameters"""
        em = self.window.theme.font_size
        
        # Title
        title = gui.Label("GaussianFeels Controls")
        title.text_color = gui.Color(0.9, 0.9, 0.9)
        self.controls_panel.add_child(title)
        
        # Training controls
        training_group = gui.CollapsableVert("Training Control", 0, gui.Margins(0, 0, 0, em))
        
        # Play/Pause button
        self.play_pause_btn = gui.Button("⏸️ Pause Training")
        self.play_pause_btn.set_on_clicked(self._on_play_pause)
        training_group.add_child(self.play_pause_btn)
        
        # Reset button
        self.reset_btn = gui.Button("🔄 Reset Training")
        self.reset_btn.set_on_clicked(self._on_reset)
        training_group.add_child(self.reset_btn)
        
        # Step button
        self.step_btn = gui.Button("⏭️ Single Step")
        self.step_btn.set_on_clicked(self._on_step)
        training_group.add_child(self.step_btn)
        
        self.controls_panel.add_child(training_group)
        
        # Learning rate controls
        lr_group = gui.CollapsableVert("Learning Rates", 0, gui.Margins(0, 0, 0, em))
        
        # Position LR
        lr_group.add_child(gui.Label("Position LR:"))
        self.position_lr_slider = gui.Slider(gui.Slider.DOUBLE)
        self.position_lr_slider.set_limits(1e-6, 1e-2)
        self.position_lr_slider.double_value = self.config.learning_rates.position
        self.position_lr_slider.set_on_value_changed(self._on_position_lr_changed)
        lr_group.add_child(self.position_lr_slider)
        
        # Rotation LR
        lr_group.add_child(gui.Label("Rotation LR:"))
        self.rotation_lr_slider = gui.Slider(gui.Slider.DOUBLE)
        self.rotation_lr_slider.set_limits(1e-5, 1e-1)
        self.rotation_lr_slider.double_value = self.config.learning_rates.rotation
        self.rotation_lr_slider.set_on_value_changed(self._on_rotation_lr_changed)
        lr_group.add_child(self.rotation_lr_slider)
        
        # Scale LR
        lr_group.add_child(gui.Label("Scale LR:"))
        self.scale_lr_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scale_lr_slider.set_limits(1e-5, 1e-1)
        self.scale_lr_slider.double_value = self.config.learning_rates.scale
        self.scale_lr_slider.set_on_value_changed(self._on_scale_lr_changed)
        lr_group.add_child(self.scale_lr_slider)
        
        # Opacity LR
        lr_group.add_child(gui.Label("Opacity LR:"))
        self.opacity_lr_slider = gui.Slider(gui.Slider.DOUBLE)
        self.opacity_lr_slider.set_limits(1e-4, 1e0)
        self.opacity_lr_slider.double_value = self.config.learning_rates.opacity
        self.opacity_lr_slider.set_on_value_changed(self._on_opacity_lr_changed)
        lr_group.add_child(self.opacity_lr_slider)
        
        self.controls_panel.add_child(lr_group)
        
        # Visualization controls
        viz_group = gui.CollapsableVert("Visualization", 0, gui.Margins(0, 0, 0, em))
        
        # Show/hide toggles
        self.show_gaussians_cb = gui.Checkbox("Show Gaussians")
        self.show_gaussians_cb.checked = True
        self.show_gaussians_cb.set_on_checked(self._on_show_gaussians)
        viz_group.add_child(self.show_gaussians_cb)
        
        self.show_tactile_cb = gui.Checkbox("Show Tactile Points")
        self.show_tactile_cb.checked = True
        self.show_tactile_cb.set_on_checked(self._on_show_tactile)
        viz_group.add_child(self.show_tactile_cb)
        
        self.show_cameras_cb = gui.Checkbox("Show Camera Frustums")
        self.show_cameras_cb.checked = True
        self.show_cameras_cb.set_on_checked(self._on_show_cameras)
        viz_group.add_child(self.show_cameras_cb)
        
        self.show_mesh_cb = gui.Checkbox("Show Mesh")
        self.show_mesh_cb.checked = False
        self.show_mesh_cb.set_on_checked(self._on_show_mesh)
        viz_group.add_child(self.show_mesh_cb)
        
        # Point size slider
        viz_group.add_child(gui.Label("Point Size:"))
        self.point_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.point_size_slider.set_limits(1.0, 10.0)
        self.point_size_slider.double_value = 3.0
        self.point_size_slider.set_on_value_changed(self._on_point_size_changed)
        viz_group.add_child(self.point_size_slider)
        
        self.controls_panel.add_child(viz_group)
    
    def _setup_status_panel(self):
        """Setup the status information panel"""
        em = self.window.theme.font_size
        
        # Status group
        status_group = gui.CollapsableVert("Training Status", 0, gui.Margins(0, 0, 0, em))
        
        # Step counter
        self.step_label = gui.Label("Step: 0")
        status_group.add_child(self.step_label)
        
        # FPS counter
        self.fps_label = gui.Label("FPS: 0.0")
        status_group.add_child(self.fps_label)
        
        # Gaussian count
        self.gaussian_count_label = gui.Label("Gaussians: 0")
        status_group.add_child(self.gaussian_count_label)
        
        # Memory usage
        self.memory_label = gui.Label("Memory: 0 MB")
        status_group.add_child(self.memory_label)
        
        # Loss values
        self.pose_loss_label = gui.Label("Pose Loss: 0.0")
        status_group.add_child(self.pose_loss_label)
        
        self.map_loss_label = gui.Label("Map Loss: 0.0")
        status_group.add_child(self.map_loss_label)
        
        self.controls_panel.add_child(status_group)
    
    def _setup_callbacks(self):
        """Setup GUI callbacks"""
        self.scene_widget.set_on_mouse(self._on_mouse)
        self.scene_widget.set_on_key(self._on_key)
        self.window.set_on_close(self._on_close)
    
    def _on_tick(self):
        """Called every frame to update the visualization"""
        if not self.running:
            return
        
        try:
            # Update status labels
            self._update_status_labels()
            
            # Update 3D visualization if needed
            if self.step_counter % self.update_interval == 0:
                self._update_3d_visualization()
        
        except Exception as e:
            print(f"Warning: GUI update failed: {e}")
    
    def _update_status_labels(self):
        """Update the status labels with current training info"""
        try:
            metrics = self.trainer.get_performance_metrics()
            
            self.step_label.text = f"Step: {metrics.get('step', 0)}"
            self.fps_label.text = f"FPS: {metrics.get('fps', 0):.1f}"
            self.gaussian_count_label.text = f"Gaussians: {metrics.get('num_gaussians', 0):,}"
            self.memory_label.text = f"Memory: {metrics.get('memory_usage_mb', 0):.1f} MB"
            self.pose_loss_label.text = f"Pose Loss: {metrics.get('pose_loss', 0):.6f}"
            self.map_loss_label.text = f"Map Loss: {metrics.get('map_loss', 0):.6f}"
        
        except Exception as e:
            # Fallback to defaults if trainer not ready
            self.step_label.text = f"Step: {self.step_counter}"
            self.fps_label.text = "FPS: --"
            self.gaussian_count_label.text = "Gaussians: --"
    
    def _update_3d_visualization(self):
        """Update the 3D scene with latest data"""
        try:
            # Update Gaussian point cloud
            if self.show_gaussians_cb.checked:
                self._update_gaussian_visualization()
            
            # Update tactile points
            if self.show_tactile_cb.checked:
                self._update_tactile_visualization()
            
            # Update camera frustums
            if self.show_cameras_cb.checked:
                self._update_camera_frustums()
                
            # Update reconstructed mesh
            if self.show_mesh_cb.checked:
                self._update_mesh_visualization()
        
        except Exception as e:
            print(f"Warning: 3D visualization update failed: {e}")
    
    def _update_gaussian_visualization(self):
        """Update the Gaussian point cloud visualization"""
        try:
            # Get Gaussian data from trainer
            if hasattr(self.trainer, 'gaussian_field'):
                positions = self.trainer.gaussian_field.get_positions()
                colors = self.trainer.gaussian_field.get_colors()
                
                if positions is not None and len(positions) > 0:
                    # Convert to numpy
                    if hasattr(positions, 'cpu'):
                        positions = positions.cpu().numpy()
                    if hasattr(colors, 'cpu'):
                        colors = colors.cpu().numpy()
                    
                    # Create/update point cloud
                    gaussian_cloud = o3d.geometry.PointCloud()
                    gaussian_cloud.points = o3d.utility.Vector3dVector(positions)
                    gaussian_cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
                    
                    # Add to scene
                    self.scene.remove_geometry("gaussians")
                    self.material.point_size = self.point_size_slider.double_value
                    self.scene.add_geometry("gaussians", gaussian_cloud, self.material)
        
        except Exception as e:
            print(f"Failed to update Gaussian visualization: {e}")
    
    def _update_tactile_visualization(self):
        """Update tactile point visualization"""
        try:
            # Get tactile data from trainer
            if hasattr(self.trainer, 'current_tactile_points'):
                tactile_points = self.trainer.current_tactile_points
                if tactile_points is not None and len(tactile_points) > 0:
                    # Convert to numpy
                    if hasattr(tactile_points, 'cpu'):
                        tactile_points = tactile_points.cpu().numpy()
                    
                    # Create tactile point cloud (red color)
                    tactile_cloud = o3d.geometry.PointCloud()
                    tactile_cloud.points = o3d.utility.Vector3dVector(tactile_points)
                    tactile_colors = np.tile([1.0, 0.0, 0.0], (len(tactile_points), 1))  # Red
                    tactile_cloud.colors = o3d.utility.Vector3dVector(tactile_colors)
                    
                    # Add to scene
                    self.scene.remove_geometry("tactile")
                    tactile_material = rendering.MaterialRecord()
                    tactile_material.shader = "defaultUnlit"
                    tactile_material.point_size = self.point_size_slider.double_value * 1.5
                    self.scene.add_geometry("tactile", tactile_cloud, tactile_material)
        
        except Exception as e:
            print(f"Failed to update tactile visualization: {e}")
    
    def _update_camera_frustums(self):
        """Update camera frustum visualization"""
        try:
            # This would show camera poses as frustums
            # Implementation depends on camera pose data from trainer
            pass
        except Exception as e:
            print(f"Failed to update camera frustums: {e}")
    
    def _update_mesh_visualization(self):
        """Update reconstructed mesh visualization"""
        try:
            # Get mesh from trainer if available
            if hasattr(self.trainer, 'get_reconstructed_mesh'):
                mesh = self.trainer.get_reconstructed_mesh()
                if mesh is not None:
                    self.scene.remove_geometry("mesh")
                    mesh_material = rendering.MaterialRecord()
                    mesh_material.shader = "defaultLit"
                    mesh_material.albedo_img = None
                    self.scene.add_geometry("mesh", mesh, mesh_material)
        
        except Exception as e:
            print(f"Failed to update mesh visualization: {e}")
    
    # GUI Event Handlers
    def _on_play_pause(self):
        """Toggle training play/pause"""
        self.training_running = not self.training_running
        if hasattr(self.trainer, 'set_training_active'):
            self.trainer.set_training_active(self.training_running)
        
        self.play_pause_btn.text = "▶️ Resume Training" if not self.training_running else "⏸️ Pause Training"
    
    def _on_reset(self):
        """Reset training"""
        if hasattr(self.trainer, 'reset_training'):
            self.trainer.reset_training()
        self.step_counter = 0
    
    def _on_step(self):
        """Perform single training step"""
        if hasattr(self.trainer, 'single_step'):
            self.trainer.single_step()
        self.step_counter += 1
    
    def _on_position_lr_changed(self, value):
        """Position learning rate changed"""
        if hasattr(self.trainer, 'update_learning_rates'):
            self.trainer.update_learning_rates(position=value)
    
    def _on_rotation_lr_changed(self, value):
        """Rotation learning rate changed"""
        if hasattr(self.trainer, 'update_learning_rates'):
            self.trainer.update_learning_rates(rotation=value)
    
    def _on_scale_lr_changed(self, value):
        """Scale learning rate changed"""
        if hasattr(self.trainer, 'update_learning_rates'):
            self.trainer.update_learning_rates(scale=value)
    
    def _on_opacity_lr_changed(self, value):
        """Opacity learning rate changed"""
        if hasattr(self.trainer, 'update_learning_rates'):
            self.trainer.update_learning_rates(opacity=value)
    
    def _on_show_gaussians(self, checked):
        """Toggle Gaussian visualization"""
        if not checked:
            self.scene.remove_geometry("gaussians")
    
    def _on_show_tactile(self, checked):
        """Toggle tactile point visualization"""
        if not checked:
            self.scene.remove_geometry("tactile")
    
    def _on_show_cameras(self, checked):
        """Toggle camera frustum visualization"""
        if not checked:
            self.scene.remove_geometry("cameras")
    
    def _on_show_mesh(self, checked):
        """Toggle mesh visualization"""
        if not checked:
            self.scene.remove_geometry("mesh")
    
    def _on_point_size_changed(self, value):
        """Point size changed"""
        self.material.point_size = value
    
    def _on_mouse(self, event):
        """Handle mouse events"""
        return gui.Widget.EventCallbackResult.HANDLED
    
    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == gui.KeyName.SPACE and event.type == gui.KeyEvent.Type.DOWN:
            self._on_play_pause()
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.key == gui.KeyName.R and event.type == gui.KeyEvent.Type.DOWN:
            self._on_reset()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _on_close(self):
        """Handle window close"""
        self.running = False
        return True
    
    def update(self):
        """Update the viewer (called from main thread)"""
        if not self.running:
            return
        
        self.step_counter += 1
    
    def stop(self):
        """Stop Open3D viewer"""
        self.running = False
        if self.window:
            try:
                gui.Application.instance.quit()
            except:
                pass

class WebViewer(BaseViewer):
    """Web-based viewer using comprehensive WebViewerManager"""
    
    def __init__(self, config: GaussianFeelsConfig, trainer: 'GaussianSplattingTrainer'):
        super().__init__(config, trainer)
        self.web_manager = None
        self.server_thread = None
        
    def start(self):
        """Start web server"""
        if self.running:
            return
        
        try:
            from .web_viewer import WebViewerManager
            print("⠋ Starting comprehensive web viewer...")
            
            self.web_manager = WebViewerManager(self.config, self.trainer)
            self.server_thread = self.web_manager.start_background()
            
            self.running = True
            print("✅ Web viewer started successfully")
            
        except ImportError as e:
            print(f"❌ Error: Web dependencies not available. Install with: pip install fastapi uvicorn websockets")
            print(f"    Details: {e}")
            raise
    
    def update(self):
        """Update connected clients with latest data"""
        if self.web_manager and self.running:
            # Update is handled automatically by the WebViewerManager via WebSockets
            # and periodic polling from the frontend
            pass
    
    def stop(self):
        """Stop web server"""
        self.running = False
        if self.web_manager:
            print("🛑 Stopping web viewer")
            # Web server will stop when the main process ends

class InteractiveGUI:
    """Main GUI class combining Open3D viewer with comprehensive controls"""
    
    def __init__(self, config: GaussianFeelsConfig, trainer: 'GaussianSplattingTrainer'):
        self.config = config
        self.trainer = trainer
        self.viewer = Open3DViewer(config, trainer)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Training state
        self.training_active = True
        self.auto_update = True
        
    def start(self):
        """Start the interactive GUI"""
        print("🚀 Starting GaussianFeels Interactive GUI...")
        self.viewer.start()
        
    def update(self, step: int):
        """Update GUI with training step"""
        if not self.auto_update:
            return
            
        # Update performance metrics
        self.performance_monitor.update_step(step)
        
        # Update viewer
        self.viewer.update()
        
    def stop(self):
        """Stop the GUI"""
        print("🛑 Stopping Interactive GUI...")
        self.viewer.stop()
        
    def save_screenshot(self, filepath: str):
        """Save a screenshot of the current view"""
        if self.viewer.window:
            try:
                img = self.viewer.scene_widget.render_to_image()
                o3d.io.write_image(filepath, img, quality=95)
                print(f"📸 Screenshot saved: {filepath}")
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

class PerformanceMonitor:
    """Monitor training performance metrics"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.step_times = []
        self.loss_values = []
        self.memory_usage = []
        self.start_time = time.time()
        
    def update_step(self, step: int):
        """Update performance metrics for current step"""
        current_time = time.time()
        
        # Calculate step time (FPS)
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1]
            fps = 1.0 / step_time if step_time > 0 else 0
        else:
            fps = 0
            
        self.step_times.append(current_time)
        
        # Keep only recent history
        if len(self.step_times) > self.history_length:
            self.step_times.pop(0)
            
    def get_metrics(self) -> dict:
        """Get current performance metrics"""
        if len(self.step_times) < 2:
            return {'fps': 0, 'avg_fps': 0, 'runtime': 0}
            
        current_time = time.time()
        recent_times = self.step_times[-10:]  # Last 10 steps
        
        # Calculate FPS
        if len(recent_times) >= 2:
            time_diffs = [recent_times[i] - recent_times[i-1] for i in range(1, len(recent_times))]
            avg_step_time = sum(time_diffs) / len(time_diffs)
            fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
        else:
            fps = 0
            
        return {
            'fps': fps,
            'avg_fps': fps,
            'runtime': current_time - self.start_time,
            'total_steps': len(self.step_times)
        }

class ViewerManager:
    """Manages different types of viewers based on configuration"""
    
    def __init__(self, config: GaussianFeelsConfig, trainer: 'GaussianSplattingTrainer'):
        self.config = config
        self.trainer = trainer
        self.viewer: Optional[BaseViewer] = None
        
    def start(self):
        """Start the appropriate viewer based on configuration"""
        viewer_type = getattr(self.config.viewer, 'type', 'none')
        
        if viewer_type == 'none':
            self.viewer = HeadlessViewer(self.config, self.trainer)
        elif viewer_type == 'open3d':
            # Use new interactive GUI for Open3D
            self.viewer = InteractiveGUI(self.config, self.trainer)
        elif viewer_type == 'web':
            self.viewer = WebViewer(self.config, self.trainer)
        else:
            print(f"Unknown viewer type: {viewer_type}, using headless")
            self.viewer = HeadlessViewer(self.config, self.trainer)
        
        self.viewer.start()
    
    def update(self, step: int = 0):
        """Update the viewer"""
        if self.viewer:
            if hasattr(self.viewer, 'update') and callable(self.viewer.update):
                if hasattr(self.viewer, 'step_counter'):
                    self.viewer.update(step)
                else:
                    self.viewer.update()
    
    def stop(self):
        """Stop the viewer"""
        if self.viewer:
            self.viewer.stop()
    
    @property
    def is_running(self) -> bool:
        """Check if viewer is running"""
        return self.viewer is not None and getattr(self.viewer, 'running', False)