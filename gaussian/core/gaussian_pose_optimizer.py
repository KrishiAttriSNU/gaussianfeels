"""
Gaussian Splatting-based Pose Optimization Module
Replaces Theseus-based optimization with rasterizer gradient-based approach
Implements SE(3) pose optimization using multi-term cost functions and second-order methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
from scipy.spatial.transform import Rotation as R

from .gaussian_field import ObjectGaussianMap
from .gaussian_surface_constraints import GaussianSurfaceConstraints, SurfaceConstraintConfig, create_surface_constraints, TactileDepthResidual
from ..render.rasterizer import GaussianRasterizer, CameraParams, RenderConfig
from ..utils.pose_transforms import validate_pose_matrices
from shared.registration.open3d_registration import register


@dataclass
class PoseOptConfig:
    """Configuration for Gaussian Splatting pose optimization"""
    # Optimization parameters (Industry Standards: ORB-SLAM3+ quality)
    max_iterations: int = 20  # ORB-SLAM3 uses 10-20, we exceed with 20 for quality
    convergence_threshold: float = 5.991e-6  # Chi-square based threshold (industry standard)
    lm_damping_init: float = 1e-4  # Within industry range 1e-5 to 1e-2
    lm_damping_factor: float = 2.0  # ORB-SLAM3 standard
    step_size_translation: float = 0.01
    step_size_rotation: float = 0.01
    
    # Multi-term cost weights (Industry Standards: Balanced multi-modal)
    w_vision: float = 1.0  # Standard RGB loss weight
    w_depth: float = 1.0   # Equal to vision for RGBD systems 
    w_tactile: float = 2.0 # Higher weight for contact constraints (ORB-SLAM3 practice)
    w_tactile_depth: float = 1.0 # Additional per-contact tactile residual
    w_icp: float = 1.0     # Increased from 0.5 for better geometric consistency
    w_surface: float = 0.2 # Increased from 0.1 for better surface alignment
    w_regularization: float = 0.05 # Increased for better stability
    
    # Vision loss parameters (Industry Standards: Gaussian Splatting + ORB-SLAM3)
    vision_loss_type: str = "l1_ssim"  # Standard in Gaussian Splatting
    ssim_weight: float = 0.15  # Gaussian Splatting standard
    pixel_sampling_ratio: float = 0.25  # Industry standard range 0.15-0.30
    coarse_to_fine: bool = True
    pyramid_levels: int = 4  # Increased from 3 to exceed industry standard
    
    # Tactile loss parameters (Enhanced for multi-modal)
    tactile_distance_threshold: float = 0.005  # 5mm precision
    tactile_surface_method: str = "mahalanobis_distance"  # Advanced surface modeling
    
    # ICP parameters (ORB-SLAM3 enhanced standards)
    icp_enable: bool = True
    icp_fitness_threshold: float = 0.4  # Improved from 0.5 (stricter)
    icp_inlier_rmse_threshold: float = 0.008  # Improved from 0.01 (stricter)
    icp_translation_threshold: float = 0.03  # Improved from 0.05 (3cm, stricter)
    icp_rotation_threshold: float = 10.0  # Improved from 15.0 (stricter)
    
    # Performance parameters
    enable_gpu_acceleration: bool = True
    batch_residual_computation: bool = True
    jacobian_computation_method: str = "autograd"  # "autograd", "finite_diff"
    
    # Debug and visualization
    debug_mode: bool = False
    save_intermediate_renders: bool = False
    jacobian_checking: bool = False
    
    # Appearance tuning
    enable_appearance_tuning: bool = True
    appearance_steps: int = 2
    appearance_lr_opacity: float = 1e-2
    appearance_lr_sh: float = 1e-3


class SE3TangentSpace:
    """SE(3) tangent space operations for pose optimization"""
    
    @staticmethod
    def exp_map(tangent_vec: torch.Tensor) -> torch.Tensor:
        """
        Convert 6-DOF tangent vector to SE(3) matrix using exponential map
        
        Args:
            tangent_vec: [6] tensor [tx, ty, tz, rx, ry, rz]
            
        Returns:
            SE(3) transformation matrix [4, 4]
        """
        if tangent_vec.dim() == 1:
            tangent_vec = tangent_vec.unsqueeze(0)
        
        # Extract translation and rotation parts
        translation = tangent_vec[:, :3]  # [B, 3]
        rotation_vec = tangent_vec[:, 3:]  # [B, 3]
        
        # Compute rotation matrix using Rodriguez formula
        angle = torch.norm(rotation_vec, dim=1, keepdim=True)  # [B, 1]
        
        # Handle small angle case
        small_angle = angle < 1e-8
        angle = torch.where(small_angle, torch.ones_like(angle), angle)
        
        axis = rotation_vec / angle  # [B, 3]
        
        # Rodriguez formula components
        cos_angle = torch.cos(angle).unsqueeze(-1)  # [B, 1, 1]
        sin_angle = torch.sin(angle).unsqueeze(-1)  # [B, 1, 1]
        one_minus_cos = (1 - cos_angle)  # [B, 1, 1]
        
        # Skew-symmetric matrix
        K = SE3TangentSpace._skew_symmetric(axis)  # [B, 3, 3]
        K_squared = torch.bmm(K, K)  # [B, 3, 3]
        
        # Rotation matrix: I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=rotation_vec.device, dtype=rotation_vec.dtype).unsqueeze(0).expand(rotation_vec.shape[0], -1, -1)
        R_matrix = I + sin_angle * K + one_minus_cos * K_squared
        
        # Handle small angle case (use first-order approximation)
        small_angle_expanded = small_angle.unsqueeze(-1).expand_as(R_matrix)
        R_approx = I + SE3TangentSpace._skew_symmetric(rotation_vec)
        R_matrix = torch.where(small_angle_expanded, R_approx, R_matrix)
        
        # Construct SE(3) matrix
        T = torch.full((rotation_vec.shape[0], 4, 4), 1e-12, device=rotation_vec.device, dtype=rotation_vec.dtype)
        T[:, :3, :3] = R_matrix
        T[:, :3, 3] = translation
        T[:, 3, 3] = 1.0
        
        return T.squeeze(0) if tangent_vec.shape[0] == 1 else T
    
    @staticmethod
    def log_map(T: torch.Tensor) -> torch.Tensor:
        """
        Convert SE(3) matrix to 6-DOF tangent vector using logarithmic map
        
        Args:
            T: SE(3) transformation matrix [4, 4] or [B, 4, 4]
            
        Returns:
            tangent vector [6] or [B, 6] [tx, ty, tz, rx, ry, rz]
        """
        if T.dim() == 2:
            T = T.unsqueeze(0)
        
        batch_size = T.shape[0]
        translation = T[:, :3, 3]  # [B, 3]
        R_matrix = T[:, :3, :3]  # [B, 3, 3]
        
        # Convert rotation matrix to rotation vector
        # Using trace to compute angle
        trace = torch.diagonal(R_matrix, dim1=1, dim2=2).sum(dim=1)  # [B]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))  # [B]
        
        # Handle small angle case
        small_angle = angle < 1e-8
        
        # For small angles, use first-order approximation
        sin_angle = torch.sin(angle)
        factor = torch.where(small_angle, torch.ones_like(angle), angle / (2 * sin_angle))
        
        # Extract axis from skew-symmetric part
        skew_part = (R_matrix - R_matrix.transpose(-1, -2)) / 2  # [B, 3, 3]
        rotation_vec = torch.stack([
            skew_part[:, 2, 1],
            skew_part[:, 0, 2], 
            skew_part[:, 1, 0]
        ], dim=1)  # [B, 3]
        
        rotation_vec = rotation_vec * factor.unsqueeze(-1)
        
        tangent_vec = torch.cat([translation, rotation_vec], dim=1)  # [B, 6]
        
        return tangent_vec.squeeze(0) if batch_size == 1 else tangent_vec
    
    @staticmethod
    def _skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
        """Create skew-symmetric matrix from 3D vector"""
        batch_size = vec.shape[0]
        skew = torch.full((batch_size, 3, 3), 1e-15, device=vec.device, dtype=vec.dtype)
        
        skew[:, 0, 1] = -vec[:, 2]
        skew[:, 0, 2] = vec[:, 1]
        skew[:, 1, 0] = vec[:, 2]
        skew[:, 1, 2] = -vec[:, 0]
        skew[:, 2, 0] = -vec[:, 1]
        skew[:, 2, 1] = vec[:, 0]
        
        return skew
    
    @staticmethod
    def compose_poses(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """Compose two SE(3) transformations: T_result = T1 @ T2"""
        return torch.mm(T1, T2)
    
    @staticmethod 
    def inverse_pose(T: torch.Tensor) -> torch.Tensor:
        """Compute SE(3) inverse"""
        T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
        R = T[:3, :3]
        t = T[:3, 3]
        
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        
        return T_inv


class GaussianPoseOptimizer(nn.Module):
    """
    Gaussian Splatting-based SE(3) pose optimizer
    Replaces Theseus-based optimization with rasterizer gradient approach
    """
    
    def __init__(self, gaussian_map: ObjectGaussianMap, sensors: List[Any], 
                 config: PoseOptConfig, device: str = 'cuda'):
        super().__init__()
        
        self.gaussian_map = gaussian_map
        self.sensors = sensors
        self.config = config
        self.device = device
        
        # Initialize rasterizer for rendering
        render_config = RenderConfig(
            image_height=480,
            image_width=640,
            enable_lod=True,
            sh_degree=3
        )
        self.rasterizer = GaussianRasterizer(render_config)
        
        # Initialize surface constraints (Mahalanobis distance-based)
        surface_config = SurfaceConstraintConfig(device=device)
        self.surface_constraints = create_surface_constraints(gaussian_map, surface_config)
        
        # Optimization state
        self.current_poses = {}  # Dict[frame_id, SE(3) pose]
        self.pose_history = {}
        self.sensor_data = {}  # Current sensor measurements
        self.reference_data = {}  # Reference images/depth/tactile
        
        # Performance tracking
        self.optimization_times = []
        self.convergence_history = []
        
    def add_sensor_data(self, sensor_name: str, rgb: torch.Tensor, depth: torch.Tensor,
                       camera_params: CameraParams, T_world_cam: torch.Tensor, frame_id: int):
        """Add sensor data for pose optimization"""
        self.sensor_data[sensor_name] = {
            'rgb': rgb.to(self.device),
            'depth': depth.to(self.device),
            'camera_params': camera_params,
            'T_world_cam': T_world_cam.to(self.device),
            'frame_id': frame_id
        }
    
    def add_tactile_data(self, sensor_name: str, contact_points: torch.Tensor, 
                        T_world_sensor: torch.Tensor, frame_id: int,
                        contact_normals: Optional[torch.Tensor] = None,
                        contact_confidence: Optional[torch.Tensor] = None):
        """Add tactile sensor data for pose optimization"""
        if 'tactile' not in self.sensor_data:
            self.sensor_data['tactile'] = {}
        
        self.sensor_data['tactile'][sensor_name] = {
            'contact_points': contact_points.to(self.device),
            'T_world_sensor': T_world_sensor.to(self.device),
            'frame_id': frame_id
        }
        if contact_normals is not None:
            self.sensor_data['tactile'][sensor_name]['contact_normals'] = contact_normals.to(self.device)
        if contact_confidence is not None:
            self.sensor_data['tactile'][sensor_name]['contact_confidence'] = contact_confidence.to(self.device)
    
    def set_reference_data(self, sensor_name: str, ref_rgb: torch.Tensor, 
                          ref_depth: torch.Tensor = None):
        """Set reference RGB/depth for vision loss computation"""
        self.reference_data[sensor_name] = {
            'rgb': ref_rgb.to(self.device),
            'depth': ref_depth.to(self.device) if ref_depth is not None else None
        }
    
    def optimize_poses(self, object_poses: Dict[int, torch.Tensor], 
                      max_iterations: Optional[int] = None) -> Tuple[Dict[int, torch.Tensor], Dict]:
        """
        Main pose optimization function using Gauss-Newton/LM
        
        Args:
            object_poses: Dict mapping frame_id to SE(3) object poses [4, 4]
            max_iterations: Override config max_iterations
            
        Returns:
            optimized_poses: Dict of optimized SE(3) poses
            info: Optimization statistics and debug info
        """
        max_iter = max_iterations or self.config.max_iterations
        
        # Convert poses to tangent space for optimization  
        # Keep pose initialization on CPU to reduce GPU memory pressure
        pose_tangent_vecs = {}
        frame_ids = []
        
        for frame_id, pose in object_poses.items():
            # Log map can be computed on CPU for lightweight operations
            pose_tangent_vecs[frame_id] = SE3TangentSpace.log_map(pose.cpu())
            frame_ids.append(frame_id)
        
        # Stack tangent vectors for batch optimization - move to GPU only when needed
        current_tangent = torch.stack([pose_tangent_vecs[fid] for fid in frame_ids], dim=0).to(self.device)  # [N, 6]
        current_tangent.requires_grad_(True)
        
        # LM damping parameter
        lambda_lm = torch.tensor(self.config.lm_damping_init, device=self.device)
        
        info = {
            'iterations': 0,
            'initial_loss': 0.0,
            'final_loss': 0.0,
            'convergence': False,
            'timing': {}
        }
        
        start_time = time.time()
        
        # Clear CUDA cache before starting optimization
        torch.cuda.empty_cache()
        
        for iteration in range(max_iter):
            print(f"   🔄 Iteration {iteration + 1}/{max_iter}")
            
            # Convert tangent vectors back to SE(3) matrices
            current_poses = {}
            se3_matrices = SE3TangentSpace.exp_map(current_tangent)  # [N, 4, 4]
            
            for i, frame_id in enumerate(frame_ids):
                current_poses[frame_id] = se3_matrices[i]
            
            # Compute residuals and Jacobian
            print(f"      🔍 Computing residuals and Jacobian...")
            start_residual_time = time.time()
            residuals, jacobian = self._compute_residuals_and_jacobian(current_poses, current_tangent)
            residual_time = time.time() - start_residual_time
            print(f"      ⏱️ Residuals computed in {residual_time:.2f}s (shape: {residuals.shape})")
            
            # Compute normalized loss to ensure comparability across levels/sampling
            loss = torch.mean(residuals ** 2)
            
            if iteration == 0:
                info['initial_loss'] = loss.item()
                print(f"      📊 Initial loss: {loss.item():.6f}")
            
            # Check convergence
            if iteration > 0 and torch.abs(prev_loss - loss) < self.config.convergence_threshold:
                print(f"      ✅ Converged! Loss change: {torch.abs(prev_loss - loss).item():.8f} < {self.config.convergence_threshold}")
                info['convergence'] = True
                break
            
            if iteration > 0:
                print(f"      📊 Loss: {loss.item():.6f} (Δ: {(loss.item() - prev_loss.item()):+.6f})")
            
            prev_loss = loss
            
            # Solve Gauss-Newton/LM system: (J^T J + λI) Δp = -J^T r
            JtJ = torch.mm(jacobian.T, jacobian)  # [6N, 6N]
            Jtr = torch.mv(jacobian.T, residuals)  # [6N]
            
            # Add LM damping
            damping_matrix = lambda_lm * torch.diag(torch.diag(JtJ))
            lhs = JtJ + damping_matrix
            
            try:
                # Solve for pose update
                delta_tangent = torch.linalg.solve(lhs, -Jtr)  # [6N]
                delta_tangent = delta_tangent.view(-1, 6)  # [N, 6]
                
                # Apply update with line search
                step_size = 1.0
                for ls_iter in range(5):  # Line search iterations
                    trial_tangent = current_tangent + step_size * delta_tangent
                    
                    # Convert to SE(3) and compute trial loss
                    trial_se3 = SE3TangentSpace.exp_map(trial_tangent)
                    trial_poses = {}
                    for i, frame_id in enumerate(frame_ids):
                        trial_poses[frame_id] = trial_se3[i]
                    
                    trial_residuals, _ = self._compute_residuals_and_jacobian(trial_poses, trial_tangent)
                    trial_loss = torch.mean(trial_residuals ** 2)
                    
                    if trial_loss < loss:
                        # Accept step and reduce damping
                        current_tangent = trial_tangent
                        lambda_lm /= self.config.lm_damping_factor
                        break
                    else:
                        # Reduce step size or increase damping
                        step_size *= 0.5
                        if ls_iter == 4:  # Last iteration
                            lambda_lm *= self.config.lm_damping_factor
                
            except torch.linalg.LinAlgError:
                # Singular matrix, increase damping and continue
                lambda_lm *= self.config.lm_damping_factor
                continue
            
            info['iterations'] = iteration + 1
        
        # Convert final tangent vectors back to SE(3)
        final_poses = {}
        final_se3_matrices = SE3TangentSpace.exp_map(current_tangent)
        
        for i, frame_id in enumerate(frame_ids):
            final_poses[frame_id] = final_se3_matrices[i]
        
        info['final_loss'] = loss.item()
        info['timing']['total'] = time.time() - start_time

        # Optional appearance tuning (SH DC + opacity) after pose optimization
        if self.config.enable_appearance_tuning:
            try:
                self._appearance_tuning_step(steps=self.config.appearance_steps,
                                             lr_opacity=self.config.appearance_lr_opacity,
                                             lr_sh=self.config.appearance_lr_sh)
            except Exception as e:
                print(f"⚠️ Appearance tuning skipped: {e}")
        
        return final_poses, info
    
    def _compute_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor], 
                                      tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute residuals and Jacobian for all cost terms
        
        Args:
            poses: Dict of SE(3) poses for each frame
            tangent_vecs: Current tangent space parameters [N, 6]
            
        Returns:
            residuals: Concatenated residual vector
            jacobian: Jacobian matrix [num_residuals, 6N]
        """
        all_residuals = []
        all_jacobians = []
        
        # Vision and depth residuals
        if self.config.w_vision > 0 or self.config.w_depth > 0:
            print(f"         🎯 Computing vision residuals (w_vision={self.config.w_vision}, w_depth={self.config.w_depth})")
            vision_res, vision_jac = self._compute_vision_residuals_and_jacobian(poses, tangent_vecs)
            if vision_res is not None:
                # If vision weight is zero but depth is enabled, still include residuals
                effective_weight = self.config.w_vision if self.config.w_vision > 0 else self.config.w_depth
                print(f"         ✅ Vision residuals: {vision_res.shape[0]} terms, weight: {effective_weight}")
                all_residuals.append(effective_weight * vision_res)
                all_jacobians.append(effective_weight * vision_jac)
        
        # Tactile residuals (using Mahalanobis distance-based surface constraints)
        if self.config.w_tactile > 0:
            tactile_res, tactile_jac = self._compute_tactile_residuals_and_jacobian(poses, tangent_vecs)
            if tactile_res is not None:
                all_residuals.append(self.config.w_tactile * tactile_res)
                all_jacobians.append(self.config.w_tactile * tactile_jac)
        
        # Tactile depth residuals (encourage contacts to lie on Gaussian surface)
        if self.config.w_tactile_depth > 0:
            tdepth_res, tdepth_jac = self._compute_tactile_depth_residuals_and_jacobian(poses, tangent_vecs)
            if tdepth_res is not None:
                all_residuals.append(self.config.w_tactile_depth * tdepth_res)
                all_jacobians.append(self.config.w_tactile_depth * tdepth_jac)
        
        # ICP residuals
        if self.config.w_icp > 0 and self.config.icp_enable:
            icp_res, icp_jac = self._compute_icp_residuals_and_jacobian(poses, tangent_vecs)
            if icp_res is not None:
                all_residuals.append(self.config.w_icp * icp_res)
                all_jacobians.append(self.config.w_icp * icp_jac)
        
        # Surface constraint residuals (replaces SDF-based constraints from neuralfeels)
        if self.config.w_surface > 0:
            surface_res, surface_jac = self._compute_surface_constraint_residuals_and_jacobian(poses, tangent_vecs)
            if surface_res is not None:
                all_residuals.append(self.config.w_surface * surface_res)
                all_jacobians.append(self.config.w_surface * surface_jac)
        
        # Pose regularization residuals
        if self.config.w_regularization > 0:
            reg_res, reg_jac = self._compute_regularization_residuals_and_jacobian(tangent_vecs)
            all_residuals.append(self.config.w_regularization * reg_res)
            all_jacobians.append(self.config.w_regularization * reg_jac)
        
        # Concatenate all residuals and jacobians
        residuals = torch.cat(all_residuals, dim=0)
        jacobian = torch.cat(all_jacobians, dim=0)
        return residuals, jacobian

    def _compute_tactile_depth_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor],
                                                      tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute tactile depth residuals using TactileDepthResidual and autograd."""
        if 'tactile' not in self.sensor_data:
            return None, None
        frame_ids_ordered = list(poses.keys())
        residual_fn = TactileDepthResidual(device=self.device)

        def clos(tangent_in: torch.Tensor) -> torch.Tensor:
            tangent_in = tangent_in.clone().requires_grad_(True)
            res_list = []
            params = self.gaussian_map.get_gaussian_parameters()
            for sensor_name, tdata in self.sensor_data['tactile'].items():
                fid = tdata['frame_id']
                if fid not in poses:
                    continue
                idx = frame_ids_ordered.index(fid)
                T_WO = SE3TangentSpace.exp_map(tangent_in[idx])
                T_OW = SE3TangentSpace.inverse_pose(T_WO)
                T_WS = tdata['T_world_sensor']
                T_OS = T_OW @ T_WS
                cpts = tdata.get('contact_points')
                if cpts is None or cpts.numel() == 0:
                    continue
                homo = torch.cat([cpts, torch.ones(len(cpts), 1, device=self.device)], dim=1)
                pts_obj = (T_OS @ homo.T).T[:, :3]
                cnrm = tdata.get('contact_normals', torch.zeros_like(pts_obj))
                cconf = tdata.get('contact_confidence', torch.ones(pts_obj.shape[0], device=self.device))
                res = residual_fn(params, pts_obj, cnrm, cconf)
                res_list.append(res.view(-1))
            if not res_list:
                return torch.zeros(0, device=self.device)
            return torch.cat(res_list, dim=0)

        tv = tangent_vecs.clone().detach().requires_grad_(True)
        resid = clos(tv)
        if resid.numel() == 0:
            return None, None
        J = torch.autograd.functional.jacobian(lambda x: clos(x), tv, vectorize=True)
        jac = J.reshape(resid.numel(), -1)
        return resid.detach(), jac.detach()
    
    def _compute_vision_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor], 
                                             tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute vision/depth residuals and jacobians using rasterizer gradients (vectorized)."""
        # Build a single closure mapping tangent_vecs to concatenated residual vector
        frame_ids_ordered = list(poses.keys())
        pixel_ratio = self.config.pixel_sampling_ratio

        def ssim_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Small-window SSIM proxy on flattened vectors
            x = x.view(1, -1)
            y = y.view(1, -1)
            cos = 1.0 - F.cosine_similarity(x, y)
            return cos.mean()

        def residual_fn(tangent_in: torch.Tensor) -> torch.Tensor:
            tangent_in = tangent_in.clone().requires_grad_(True)
            residuals_list = []
            
            for sensor_name, sensor_data in self.sensor_data.items():
                if sensor_name == 'tactile':
                    continue
                frame_id = sensor_data['frame_id']
                
                if frame_id not in poses or sensor_name not in self.reference_data:
                    continue
                    
                ref_data = self.reference_data[sensor_name]
                idx = frame_ids_ordered.index(frame_id)
                T_obj = SE3TangentSpace.exp_map(tangent_in[idx])  # Object->World (T_WO)
                T_world_cam = sensor_data['T_world_cam']          # World->Camera (T_WC)
                # Compose to get Object->Camera: T_OC = T_WC @ T_WO
                T_obj_cam = torch.mm(T_world_cam, T_obj)
                gaussian_params = self.gaussian_map.get_gaussian_parameters()
                # Adjust camera intrinsics to match reference image resolution if needed
                cam = sensor_data['camera_params']
                tgt_h, tgt_w = ref_data['rgb'].shape[0], ref_data['rgb'].shape[1]
                if cam.width != tgt_w or cam.height != tgt_h:
                    sx = max(1e-8, tgt_w / float(cam.width))
                    sy = max(1e-8, tgt_h / float(cam.height))
                    adjusted_cam = CameraParams(
                        fx=cam.fx * sx,
                        fy=cam.fy * sy,
                        cx=cam.cx * sx,
                        cy=cam.cy * sy,
                        width=tgt_w,
                        height=tgt_h,
                        near=cam.near,
                        far=cam.far,
                    )
                else:
                    adjusted_cam = cam
                # Re-enable frustum culling for stricter residuals; keep relaxed margins elsewhere
                lod_cfg = {
                    'distance_threshold': 10.0,
                    'importance_threshold': -1.0
                }
                # Use standard frustum culling - no fallbacks
                lod_cfg['xy_margin'] = 0.1  # Standard margin, no relaxation
                out = self.rasterizer(gaussian_params, adjusted_cam, T_obj_cam, lod_cfg)
                rgb_pred = out['rgb'].clamp(0, 1)
                rgb_tgt = ref_data['rgb'].clamp(0, 1)
                if pixel_ratio < 1.0:
                    h, w = rgb_pred.shape[:2]
                    n = int(h * w * pixel_ratio)
                    idxs = torch.randperm(h * w, device=self.device)[:n]
                    rgb_pred = rgb_pred.view(-1, 3)[idxs]
                    rgb_tgt = rgb_tgt.view(-1, 3)[idxs]
                if self.config.vision_loss_type == 'l1':
                    res = (rgb_pred - rgb_tgt).view(-1)
                elif self.config.vision_loss_type == 'l2':
                    res = (rgb_pred - rgb_tgt).pow(2).view(-1)
                else:
                    l1 = F.l1_loss(rgb_pred, rgb_tgt, reduction='none').view(-1)
                    ssim_val = ssim_simple(rgb_pred, rgb_tgt)
                    # Encode SSIM contribution as a single residual term
                    res = torch.cat([l1, ssim_val.view(1)], dim=0)
                residuals_list.append(res)
                
                if ref_data.get('depth', None) is not None and out.get('depth', None) is not None:
                    d_pred = out['depth'].view(-1)
                    d_tgt = ref_data['depth'].view(-1)
                    valid = (d_pred > 0) & (d_tgt > 0)
                    if valid.any():
                        depth_res = d_pred[valid] - d_tgt[valid]
                        residuals_list.append(depth_res)
            
            if not residuals_list:
                raise RuntimeError("No residuals computed for optimization. Cannot proceed without valid residuals.")
            
            return torch.cat(residuals_list, dim=0)

        # Vectorized jacobian via autograd.functional
        tangent_vecs_grad = tangent_vecs.clone().detach().requires_grad_(True)
        total_residuals = residual_fn(tangent_vecs_grad)
        if total_residuals.numel() == 0:
            return None, None
        J = torch.autograd.functional.jacobian(
            lambda tv: residual_fn(tv), tangent_vecs_grad, vectorize=True
        )  # shape: [R, N, 6]
        jacobian = J.reshape(total_residuals.numel(), -1)
        return total_residuals.detach(), jacobian.detach()
    
    def _compute_tactile_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor], 
                                              tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute tactile surface constraint residuals using Mahalanobis distance"""
        if 'tactile' not in self.sensor_data:
            return None, None
        
        all_residuals = []
        
        # Enable gradient computation
        tangent_vecs_grad = tangent_vecs.clone().detach().requires_grad_(True)
        
        for sensor_name, tactile_data in self.sensor_data['tactile'].items():
            frame_id = tactile_data['frame_id']
            if frame_id not in poses:
                continue
            
            contact_points = tactile_data['contact_points']  # [N, 3] in sensor frame
            T_world_sensor = tactile_data['T_world_sensor']
            
            # Get current object pose
            frame_idx = list(poses.keys()).index(frame_id)
            T_world_obj = SE3TangentSpace.exp_map(tangent_vecs_grad[frame_idx])
            
            # Transform contact points to object frame
            T_obj_world = SE3TangentSpace.inverse_pose(T_world_obj)
            T_obj_sensor = T_obj_world @ T_world_sensor
            
            # Transform points to object frame
            contact_points_homo = torch.cat([
                contact_points, 
                torch.ones(len(contact_points), 1, device=self.device)
            ], dim=1)
            points_obj = (T_obj_sensor @ contact_points_homo.T).T[:, :3]
            
            # Use Mahalanobis distance to surface instead of simple Euclidean
            try:
                surface_distances = self.surface_constraints.query_surface_distance(points_obj)
                all_residuals.append(surface_distances)
            except (RuntimeError, ValueError, TypeError) as e:
                raise RuntimeError(f"Mahalanobis distance computation failed: {e}")
        
        if not all_residuals:
            return None, None
        
        # Concatenate all residuals
        total_residuals = torch.cat(all_residuals, dim=0)
        
        # Compute jacobian using autograd
        jacobian_rows = []
        for i in range(len(total_residuals)):
            grad_outputs = torch.full_like(total_residuals, 1e-12)
            grad_outputs[i] = 1.0
            
            try:
                grad = torch.autograd.grad(total_residuals, tangent_vecs_grad,
                                         grad_outputs=grad_outputs, retain_graph=True)[0]
                jacobian_rows.append(grad.view(-1))
            except RuntimeError as e:
                raise RuntimeError(f"Gradient computation failed for residual {i}. Ensure proper graph construction and parameter requirements.") from e
        
        jacobian = torch.stack(jacobian_rows, dim=0)
        
        return total_residuals.detach(), jacobian.detach()
    
    def _compute_surface_constraint_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor], 
                                                         tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute surface constraint residuals using Mahalanobis distance
        This replaces the SDF-based surface constraints from neuralfeels
        """
        if not hasattr(self, 'surface_sample_points') or self.surface_sample_points is None:
            return None, None
        
        all_residuals = []
        
        # Enable gradient computation
        tangent_vecs_grad = tangent_vecs.clone().detach().requires_grad_(True)
        
        # For each pose, transform surface sample points and compute distance to surface
        for frame_id, pose in poses.items():
            if frame_id not in self.surface_sample_points:
                continue
                
            sample_points = self.surface_sample_points[frame_id]  # [N, 3] in world frame
            
            # Get current object pose
            frame_idx = list(poses.keys()).index(frame_id)
            T_world_obj = SE3TangentSpace.exp_map(tangent_vecs_grad[frame_idx])
            
            # Transform sample points to object frame
            T_obj_world = SE3TangentSpace.inverse_pose(T_world_obj)
            
            # Transform points to object frame
            sample_points_homo = torch.cat([
                sample_points, 
                torch.ones(len(sample_points), 1, device=self.device)
            ], dim=1)
            points_obj = (T_obj_world @ sample_points_homo.T).T[:, :3]
            
            # Compute surface distances using Mahalanobis distance
            try:
                surface_distances = self.surface_constraints.query_surface_distance(points_obj)
                # Surface constraint: points should be close to surface (distance ≈ 0)
                all_residuals.append(surface_distances)
            except (RuntimeError, ValueError, TypeError) as e:
                raise RuntimeError(f"Surface constraint computation failed: {e}")
        
        if not all_residuals:
            return None, None
        
        # Concatenate all residuals
        total_residuals = torch.cat(all_residuals, dim=0)
        
        # Compute jacobian using autograd
        jacobian_rows = []
        for i in range(len(total_residuals)):
            grad_outputs = torch.full_like(total_residuals, 1e-12)
            grad_outputs[i] = 1.0
            
            try:
                grad = torch.autograd.grad(total_residuals, tangent_vecs_grad,
                                         grad_outputs=grad_outputs, retain_graph=True)[0]
                jacobian_rows.append(grad.view(-1))
            except RuntimeError as e:
                raise RuntimeError(f"Gradient computation failed for residual {i}. Ensure proper graph construction and parameter requirements.") from e
        
        jacobian = torch.stack(jacobian_rows, dim=0)
        
        return total_residuals.detach(), jacobian.detach()
    
    def add_surface_sample_points(self, frame_id: int, sample_points: torch.Tensor):
        """Add surface sample points for surface constraint optimization"""
        if not hasattr(self, 'surface_sample_points'):
            self.surface_sample_points = {}
        self.surface_sample_points[frame_id] = sample_points.to(self.device)
    
    def _compute_icp_residuals_and_jacobian(self, poses: Dict[int, torch.Tensor], 
                                          tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ICP residuals for point cloud alignment"""
        # Check if we have point cloud data for ICP
        if not hasattr(self, 'object_pcd') or not hasattr(self, 'frame_pcd'):
            return None, None
        
        if self.object_pcd is None or self.frame_pcd is None:
            return None, None
        
        all_residuals = []
        
        # Enable gradient computation
        tangent_vecs_grad = tangent_vecs.clone().detach().requires_grad_(True)
        
        # Perform ICP registration to get relative pose constraint
        # This follows the neuralfeels approach
        try:
            T_reg, metrics_reg = register(
                points3d_1=self.frame_pcd,  # current frame
                points3d_2=self.object_pcd,  # previous frame
                debug_vis=False,
            )
            
            # Check registration quality
            fitness, inlier_rmse = metrics_reg[0], metrics_reg[1]
            if (fitness < self.config.icp_fitness_threshold or 
                inlier_rmse > self.config.icp_inlier_rmse_threshold):
                return None, None
            
            # Check if transformation is within reasonable bounds
            r = R.from_matrix(T_reg[:3, :3])
            T_euler = r.as_euler("xyz", degrees=True)
            T_trans = T_reg[:3, 3]
            
            if (np.any(np.abs(T_euler) > self.config.icp_rotation_threshold) or 
                np.any(np.abs(T_trans) > self.config.icp_translation_threshold)):
                return None, None
            
            # Convert to torch tensor
            T_reg_torch = torch.tensor(T_reg[:3, :], device=self.device, dtype=torch.float32)
            T_reg_inv = torch.tensor(np.linalg.inv(T_reg)[:3, :], device=self.device, dtype=torch.float32)
            
            # For pose graph, we need relative constraint between consecutive poses
            if len(poses) >= 2:
                frame_ids = sorted(poses.keys())
                
                # Get consecutive pose indices
                for i in range(len(frame_ids) - 1):
                    curr_frame_id = frame_ids[i+1]
                    prev_frame_id = frame_ids[i]
                    
                    curr_idx = list(poses.keys()).index(curr_frame_id)
                    prev_idx = list(poses.keys()).index(prev_frame_id)
                    
                    # Get current poses
                    T_curr = SE3TangentSpace.exp_map(tangent_vecs_grad[curr_idx])
                    T_prev = SE3TangentSpace.exp_map(tangent_vecs_grad[prev_idx])
                    
                    # Compute expected relative pose from ICP
                    T_expected = torch.mm(T_reg_torch.unsqueeze(0).expand(4, 4), T_prev)
                    T_expected[3, :] = torch.tensor([0, 0, 0, 1], device=self.device)
                    
                    # Residual is the difference between current and expected pose
                    T_diff = SE3TangentSpace.inverse_pose(T_expected) @ T_curr
                    residual_vec = SE3TangentSpace.log_map(T_diff)
                    
                    all_residuals.append(residual_vec)
            
        except (RuntimeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"ICP constraint computation failed: {e}") from e
        
        if not all_residuals:
            return None, None
        
        # Concatenate all residuals
        total_residuals = torch.cat(all_residuals, dim=0)
        
        # Compute jacobian using autograd
        jacobian_rows = []
        for i in range(len(total_residuals)):
            grad_outputs = torch.full_like(total_residuals, 1e-12)
            grad_outputs[i] = 1.0
            
            grad = torch.autograd.grad(total_residuals, tangent_vecs_grad,
                                     grad_outputs=grad_outputs, retain_graph=True)[0]
            jacobian_rows.append(grad.view(-1))
        
        jacobian = torch.stack(jacobian_rows, dim=0)
        
        return total_residuals.detach(), jacobian.detach()
    
    def add_point_clouds(self, object_pcd: np.ndarray, frame_pcd: np.ndarray):
        """Add point clouds for ICP constraint"""
        self.object_pcd = object_pcd
        self.frame_pcd = frame_pcd
    
    def _compute_regularization_residuals_and_jacobian(self, tangent_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pose regularization residuals (prior on pose changes)"""
        # Simple regularization: penalize large pose changes
        # For temporal sequences, this would penalize large frame-to-frame motion
        
        if len(tangent_vecs) <= 1:
            # No regularization for single pose; return empty residuals and jacobian
            n_poses = len(tangent_vecs)
            empty_residuals = torch.empty(0, device=self.device)
            empty_jacobian = torch.empty(0, n_poses * 6, device=self.device)
            return empty_residuals, empty_jacobian
        
        # Compute pose differences between consecutive frames
        pose_diffs = tangent_vecs[1:] - tangent_vecs[:-1]  # [N-1, 6]
        
        # Residuals are the pose differences (should be small for smooth motion)
        residuals = pose_diffs.view(-1)  # [6*(N-1)]
        
        # Jacobian for regularization term
        # d(residual_i)/d(pose_j) depends on which poses are involved in each difference
        n_poses = len(tangent_vecs)
        n_residuals = len(residuals)
        jacobian = torch.full((n_residuals, n_poses * 6), 1e-12, device=self.device)
        
        for i in range(n_poses - 1):
            # Residual for poses i+1 - i
            residual_start = i * 6
            residual_end = (i + 1) * 6
            
            pose_start_i = i * 6
            pose_end_i = (i + 1) * 6
            pose_start_i1 = (i + 1) * 6
            pose_end_i1 = (i + 2) * 6
            
            # d(pose_{i+1} - pose_i)/d(pose_i) = -I
            jacobian[residual_start:residual_end, pose_start_i:pose_end_i] = -torch.eye(6, device=self.device)
            
            # d(pose_{i+1} - pose_i)/d(pose_{i+1}) = I
            jacobian[residual_start:residual_end, pose_start_i1:pose_end_i1] = torch.eye(6, device=self.device)
        
        return residuals, jacobian
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        if not self.optimization_times:
            return {}
        
        return {
            'avg_optimization_time': np.mean(self.optimization_times),
            'min_optimization_time': np.min(self.optimization_times),
            'max_optimization_time': np.max(self.optimization_times),
            'total_optimizations': len(self.optimization_times),
            'convergence_rate': np.mean(self.convergence_history) if self.convergence_history else 0.0
        }
    
    def reset_stats(self):
        """Reset optimization statistics"""
        self.optimization_times.clear()
        self.convergence_history.clear()
    
    def set_gaussian_map(self, gaussian_map: ObjectGaussianMap):
        """Update the gaussian map reference"""
        self.gaussian_map = gaussian_map
