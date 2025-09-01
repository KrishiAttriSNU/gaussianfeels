"""
Object reconstruction utilities
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ObjectReconstructor:
    """Simple object reconstructor for point cloud accumulation"""
    
    def __init__(self):
        self.points = None
        self.colors = None
        
    def add_points(self, points, colors=None):
        """Add points to reconstruction"""
        if self.points is None:
            self.points = points
            self.colors = colors
        else:
            self.points = np.vstack([self.points, points])
            if colors is not None and self.colors is not None:
                self.colors = np.vstack([self.colors, colors])
    
    def save_point_cloud(self, path):
        """Save point cloud"""
        if self.points is None:
            return False
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
            
        return o3d.io.write_point_cloud(str(path), pcd)

def reconstruct_object_from_feelsight(data_path, output_path, max_frames=None):
    """Reconstruct object from feelsight data"""
    logger.info(f"Object reconstruction from {data_path}")
    
    # Simple reconstruction - just combine all camera points
    reconstructor = ObjectReconstructor()
    
    data_path = Path(data_path)
    
    # Find camera data files
    camera_files = list(data_path.glob("camera_*.npy"))
    if max_frames:
        camera_files = camera_files[:max_frames]
    
    logger.info(f"Found {len(camera_files)} camera files")
    
    for camera_file in camera_files:
        try:
            # Load camera data
            camera_data = np.load(camera_file, allow_pickle=True).item()
            
            if 'points' in camera_data:
                points = camera_data['points']
                colors = camera_data.get('colors', None)
                reconstructor.add_points(points, colors)
                
        except Exception as e:
            logger.warning(f"Failed to load {camera_file}: {e}")
    
    # Save result
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "simple_reconstruction.ply"
    success = reconstructor.save_point_cloud(output_file)
    
    if success:
        logger.info(f"Reconstruction saved to {output_file}")
        return {
            'success': True,
            'output_path': output_file,
            'num_points': len(reconstructor.points) if reconstructor.points is not None else 0
        }
    else:
        return {'success': False, 'error': 'Failed to save reconstruction'}