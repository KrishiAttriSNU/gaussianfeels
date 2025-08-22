"""
Rich visualization suite with dedicated viz/ directory.
Implements 3D plotting, camera pose visualization, interactive GUI, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
# Optional 3D plotting - may fail due to matplotlib version conflicts
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D_PLOTTING = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"3D plotting unavailable due to matplotlib conflict: {e}")
    Axes3D = None
    HAS_3D_PLOTTING = False
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import open3d as o3d
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GfVisualization:
    """
    Rich visualization suite.
    Implements 3D plotting, camera pose visualization, and an interactive GUI.
    """
    
    def __init__(self, output_dir: str = "viz"):
        """Initialize visualization suite"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualization settings
        self.figure_size = (12, 8)
        self.dpi = 300
        self.color_scheme = "viridis"
        
        logger.info(f"Initialized visualization suite (output: {self.output_dir})")
    
    def plot_3d_points_and_cameras(self,
                                  points: np.ndarray,
                                  camera_poses: Optional[List[np.ndarray]] = None,
                                  point_colors: Optional[np.ndarray] = None,
                                  title: str = "3D Scene Visualization") -> str:
        """
        3D plotting.
        
        Args:
            points: 3D points to visualize [N, 3]
            camera_poses: List of camera pose matrices [4, 4]
            point_colors: Colors for points [N, 3] or [N]
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        if point_colors is not None:
            if point_colors.ndim == 1:
                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                   c=point_colors, cmap=self.color_scheme, s=1)
                plt.colorbar(scatter)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=point_colors, s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='blue', s=1, alpha=0.6)
        
        # Plot camera poses
        if camera_poses is not None:
            for i, pose in enumerate(camera_poses):
                # Extract camera position
                cam_pos = pose[:3, 3]
                
                # Extract camera orientation vectors
                cam_x = pose[:3, 0] * 0.05  # Right vector (red)
                cam_y = pose[:3, 1] * 0.05  # Up vector (green)  
                cam_z = pose[:3, 2] * 0.05  # Forward vector (blue)
                
                # Plot camera position
                ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                          c='red', s=50, marker='o')
                
                # Plot camera orientation vectors
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                         cam_x[0], cam_x[1], cam_x[2], 
                         color='red', alpha=0.8, arrow_length_ratio=0.1)
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                         cam_y[0], cam_y[1], cam_y[2], 
                         color='green', alpha=0.8, arrow_length_ratio=0.1)
                ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                         cam_z[0], cam_z[1], cam_z[2], 
                         color='blue', alpha=0.8, arrow_length_ratio=0.1)
                
                # Add camera label
                ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'Cam{i}', fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Equal aspect ratio
        max_range = np.array([points[:,0].max()-points[:,0].min(),
                             points[:,1].max()-points[:,1].min(),
                             points[:,2].max()-points[:,2].min()]).max() / 2.0
        mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
        mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
        mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save plot
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D plot saved: {output_path}")
        return str(output_path)
    
    def create_interactive_3d_plot(self,
                                  points: np.ndarray,
                                  point_colors: Optional[np.ndarray] = None,
                                  title: str = "Interactive 3D Visualization") -> str:
        """
        Interactive GUI for debugging.
        
        Args:
            points: 3D points [N, 3]
            point_colors: Point colors [N]
            title: Plot title
            
        Returns:
            Path to saved HTML plot
        """
        fig = go.Figure()
        
        if point_colors is not None:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1], 
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=point_colors,
                    colorscale=self.color_scheme,
                    showscale=True
                ),
                name='Points'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2], 
                mode='markers',
                marker=dict(size=2, color='blue'),
                name='Points'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=1200,
            height=800
        )
        
        # Save interactive plot
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}_interactive.html"
        fig.write_html(output_path)
        
        logger.info(f"Interactive 3D plot saved: {output_path}")
        return str(output_path)
    
    def plot_detailed_metrics(self,
                            metrics_dict: Dict[str, Any],
                            title: str = "Detailed Metrics Analysis") -> str:
        """
        Detailed plotting utilities for metrics.
        
        Args:
            metrics_dict: Dictionary with metrics data
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        # Create subplots for different metric types
        n_metrics = len(metrics_dict)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (metric_name, metric_data) in enumerate(metrics_dict.items()):
            ax = axes[i]
            
            if isinstance(metric_data, dict):
                # Plot dictionary metrics as bar chart
                keys = list(metric_data.keys())
                values = [metric_data[k] for k in keys if isinstance(metric_data[k], (int, float))]
                valid_keys = [k for k in keys if isinstance(metric_data[k], (int, float))]
                
                if valid_keys:
                    bars = ax.bar(valid_keys, values)
                    ax.set_title(f"{metric_name}")
                    ax.set_ylabel("Value")
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    # Color bars based on value
                    for bar, value in zip(bars, values):
                        bar.set_color(plt.cm.viridis(value / max(values) if max(values) > 0 else 0))
                
            elif isinstance(metric_data, (list, np.ndarray)):
                # Plot array metrics as line plot
                ax.plot(metric_data)
                ax.set_title(f"{metric_name}")
                ax.set_ylabel("Value")
                ax.set_xlabel("Index")
                ax.grid(True, alpha=0.3)
                
            else:
                # Single value - display as text
                ax.text(0.5, 0.5, f"{metric_name}:\n{metric_data:.4f}" if isinstance(metric_data, (int, float)) else str(metric_data),
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide extra subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed metrics plot saved: {output_path}")
        return str(output_path)
    
    def visualize_parameter_gradients(self,
                                    parameter_dict: Dict[str, torch.Tensor],
                                    title: str = "Parameter Gradients") -> str:
        """
        Network parameter and gradient visualization.
        
        Args:
            parameter_dict: Dictionary with parameters and gradients
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        n_params = len(parameter_dict)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (param_name, param_tensor) in enumerate(parameter_dict.items()):
            ax = axes[i]
            
            if isinstance(param_tensor, torch.Tensor):
                param_np = param_tensor.detach().cpu().numpy()
                
                if param_np.ndim == 1:
                    # 1D parameter - line plot
                    ax.plot(param_np)
                    ax.set_title(f"{param_name} (1D)")
                    
                elif param_np.ndim == 2:
                    # 2D parameter - heatmap
                    im = ax.imshow(param_np, cmap=self.color_scheme, aspect='auto')
                    ax.set_title(f"{param_name} (2D)")
                    plt.colorbar(im, ax=ax)
                    
                else:
                    # Higher dimensional - flatten and plot histogram
                    ax.hist(param_np.flatten(), bins=50, alpha=0.7)
                    ax.set_title(f"{param_name} (Histogram)")
                    ax.set_ylabel("Frequency")
                    ax.set_xlabel("Value")
                
                # Add statistics text
                mean_val = np.mean(param_np)
                std_val = np.std(param_np)
                ax.text(0.02, 0.98, f"μ={mean_val:.4f}\nσ={std_val:.4f}", 
                       transform=ax.transAxes, va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Parameter visualization saved: {output_path}")
        return str(output_path)
    
    def create_training_analysis_dashboard(self,
                                         training_metrics: Dict[str, List[float]],
                                         title: str = "Training Analysis") -> str:
        """
        Post-hoc analysis and visualization.
        
        Args:
            training_metrics: Dictionary with training metrics over time
            title: Dashboard title
            
        Returns:
            Path to saved HTML dashboard
        """
        # Create subplots
        n_metrics = len(training_metrics)
        subplot_titles = list(training_metrics.keys())
        
        fig = make_subplots(
            rows=(n_metrics + 1) // 2, cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (metric_name, values) in enumerate(training_metrics.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines',
                    name=metric_name,
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=row, col=col
            )
            
        fig.update_layout(
            title=title,
            height=400 * ((n_metrics + 1) // 2),
            showlegend=False
        )
        
        # Update axes labels
        for i in range(n_metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.update_xaxes(title_text="Iteration", row=row, col=col)
            fig.update_yaxes(title_text="Value", row=row, col=col)
        
        # Save dashboard
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}_dashboard.html"
        fig.write_html(output_path)
        
        logger.info(f"Training analysis dashboard saved: {output_path}")
        return str(output_path)


class InteractiveVisualizationGUI:
    """
    Interactive GUI for debugging and analysis.
    """
    
    def __init__(self):
        """Initialize interactive visualization GUI"""
        self.visualizer = GfVisualization()
        
        logger.info("Initialized interactive GUI")
    
    def launch_3d_viewer(self, points: np.ndarray, **kwargs):
        """
        Launch 3D viewer for interactive debugging.
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if 'colors' in kwargs:
            pcd.colors = o3d.utility.Vector3dVector(kwargs['colors'])
        else:
            # Default blue color
            pcd.paint_uniform_color([0.0, 0.0, 1.0])
        
        # Launch viewer
        logger.info("Launching Open3D 3D viewer for interactive debugging")
        o3d.visualization.draw_geometries([pcd], window_name="GaussianFeels 3D Viewer")
    
    def create_coordinate_system_debug_view(self, 
                                          transforms: Dict[str, np.ndarray],
                                          points: Optional[np.ndarray] = None):
        """
        Coordinate system debugging and validation.
        """
        geometries = []
        
        # Add coordinate frame for each transform
        for name, transform in transforms.items():
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=transform[:3, 3]
            )
            # Apply rotation
            coord_frame.rotate(transform[:3, :3], transform[:3, 3])
            geometries.append(coord_frame)
        
        # Add points if provided
        if points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(pcd)
        
        # Launch viewer
        logger.info("Launching coordinate system debug view")
        o3d.visualization.draw_geometries(geometries, window_name="Coordinate System Debug")


if __name__ == "__main__":
    # Test the visualization suite
    print("🎨 Testing Visualization Suite")
    print("=" * 50)
    
    # Create test data
    points = np.random.randn(1000, 3) * 0.5
    colors = np.random.rand(1000)
    
    # Initialize visualization
    viz = GfVisualization()
    
    # Test 3D plotting
    plot_path = viz.plot_3d_points_and_cameras(points, point_colors=colors)
    print(f"3D plot saved: {plot_path}")
    
    # Test interactive plot
    interactive_path = viz.create_interactive_3d_plot(points, colors)
    print(f"Interactive plot saved: {interactive_path}")
    
    # Test metrics plotting
    test_metrics = {
        'accuracy': {'train': 0.95, 'val': 0.87, 'test': 0.89},
        'loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2],
        'f_score': 0.92
    }
    metrics_path = viz.plot_detailed_metrics(test_metrics)
    print(f"Metrics plot saved: {metrics_path}")
    
    print("\n✅ Visualization suite test completed!")