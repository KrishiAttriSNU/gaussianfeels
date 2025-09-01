#!/usr/bin/env python3
"""
Open3D-based registration pipeline for object-frame reconstruction.
Implements FPFH + RANSAC global registration followed by ICP refinement.
"""

import open3d as o3d
import numpy as np
from typing import Tuple


def preprocess_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.003) -> o3d.geometry.PointCloud:
    """
    Preprocess point cloud: downsample and estimate normals
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling (default: 3mm)
        
    Returns:
        Preprocessed point cloud
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
    )
    return pcd_down


def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.003) -> o3d.pipelines.registration.Feature:
    """
    Compute Fast Point Feature Histograms (FPFH) for point cloud
    
    Args:
        pcd: Input point cloud (should have normals)
        voxel_size: Voxel size for radius calculation
        
    Returns:
        FPFH features
    """
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )


def coarse_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                       source_fpfh: o3d.pipelines.registration.Feature, 
                       target_fpfh: o3d.pipelines.registration.Feature,
                       voxel_size: float = 0.003) -> o3d.pipelines.registration.RegistrationResult:
    """
    RANSAC-based global registration using FPFH features
    
    Args:
        source: Source point cloud
        target: Target point cloud  
        source_fpfh: FPFH features for source
        target_fpfh: FPFH features for target
        voxel_size: Voxel size for distance threshold
        
    Returns:
        Registration result
    """
    distance_threshold = voxel_size * 1.5  # ≈4 mm tolerance
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, 
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 0.999)
    )
    
    return result


def refine_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
               init_T: np.ndarray, voxel_size: float = 0.003) -> o3d.pipelines.registration.RegistrationResult:
    """
    ICP refinement using point-to-plane estimation
    
    Args:
        source: Source point cloud
        target: Target point cloud
        init_T: Initial transformation from global registration
        voxel_size: Voxel size for distance threshold
        
    Returns:
        Refined registration result
    """
    distance_threshold = voxel_size * 1.0  # Tighter tolerance for ICP
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    
    return result


def icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_init=np.eye(4),
    mcd=0.01,
    max_iter=15,
):
    """Point to point ICP registration"""
    result = o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=mcd,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        ),
    )
    transformation = result.transformation
    metrics = [result.fitness, result.inlier_rmse, result.correspondence_set]
    return transformation, metrics


def register(
    points3d_1,
    points3d_2,
    T_init=np.eye(4),
    debug_vis=False,
):
    """Register two point clouds using ICP"""
    # Accept either Open3D point clouds or raw Nx3 numpy arrays
    def _to_o3d_pcd(x):
        if isinstance(x, o3d.geometry.PointCloud):
            return x
        arr = np.asarray(x)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected Nx3 array for points, got shape {arr.shape}")
        pcd = o3d.geometry.PointCloud()
        # Open3D expects float64 or float32
        pcd.points = o3d.utility.Vector3dVector(arr.astype(np.float64, copy=False))
        return pcd

    cloud_1, cloud_2 = (_to_o3d_pcd(points3d_1), _to_o3d_pcd(points3d_2))
    T_reg, metrics_reg = icp(source=cloud_1, target=cloud_2, T_init=T_init)
    return T_reg, metrics_reg


def register_to_reference(points: np.ndarray, colors: np.ndarray,
                         ref_pcd: o3d.geometry.PointCloud, ref_fpfh: o3d.pipelines.registration.Feature,
                         voxel_size: float = 0.003) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Register a point cloud to a reference using global + ICP registration
    
    Args:
        points: Input points (N, 3)
        colors: Input colors (N, 3)
        ref_pcd: Reference point cloud (preprocessed)
        ref_fpfh: Reference FPFH features
        voxel_size: Voxel size for registration
        
    Returns:
        Tuple of (aligned_points, aligned_colors, registration_info)
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Preprocess
    pcd_down = preprocess_cloud(pcd, voxel_size)
    
    # Compute features
    pcd_fpfh = compute_fpfh(pcd_down, voxel_size)
    
    # Global registration
    ransac_result = coarse_registration(pcd_down, ref_pcd, pcd_fpfh, ref_fpfh, voxel_size)
    
    # ICP refinement
    icp_result = refine_icp(pcd_down, ref_pcd, ransac_result.transformation, voxel_size)
    
    # Apply transformation to original full-resolution cloud
    pcd_aligned = pcd.transform(icp_result.transformation)
    
    # Extract aligned points and colors
    aligned_points = np.asarray(pcd_aligned.points)
    aligned_colors = np.asarray(pcd_aligned.colors)
    
    # Registration info
    reg_info = {
        'ransac_fitness': ransac_result.fitness,
        'ransac_rmse': ransac_result.inlier_rmse,
        'icp_fitness': icp_result.fitness,
        'icp_rmse': icp_result.inlier_rmse,
        'transformation': icp_result.transformation
    }
    
    return aligned_points, aligned_colors, reg_info


def is_registration_valid(reg_info: dict, max_rmse: float = 0.003, min_fitness: float = 0.45) -> bool:
    """
    Check if registration meets quality thresholds
    
    Args:
        reg_info: Registration information dictionary
        max_rmse: Maximum allowed RMSE (meters)
        min_fitness: Minimum required fitness
        
    Returns:
        True if registration is valid
    """
    return (reg_info['icp_rmse'] <= max_rmse and 
            reg_info['icp_fitness'] >= min_fitness)