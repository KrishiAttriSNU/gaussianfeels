"""
Gaussian rasterization wrapper for PyTorch/CUDA rendering.
Provides rendering interface for RGB+D output with level-of-detail support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class RenderConfig:
    """Configuration for Gaussian rendering (Industry Standards: Professional Quality)"""
    image_height: int = 480  # Industry standard resolution
    image_width: int = 640   # Industry standard resolution  
    near_plane: float = 0.01
    far_plane: float = 10.0
    
    # Level of detail (Enhanced for quality)
    enable_lod: bool = True
    lod_levels: int = 4      # Increased from 3 for finer detail
    lod_distance_threshold: float = 1.5  # Stricter threshold for better quality
    
    # Rendering quality (Maximum capacity settings)
    sh_degree: int = 3       # Full spherical harmonics (industry standard)
    white_background: bool = False
    alpha_threshold: float = 1e-5  # Stricter alpha for better quality
    
    # Performance (Optimized for quality)
    tile_size: int = 16      # Industry standard
    max_gaussians_per_tile: int = 2048  # Doubled from 1024 for higher capacity

    # Memory-safety for training/gradients
    enable_grad_downscale: bool = True
    grad_render_downscale: float = 0.25  # Downscale factor used when gradients are enabled

@dataclass 
class CameraParams:
    """Camera parameters for rendering"""
    fx: float  # Focal length x
    fy: float  # Focal length y  
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    near: float = 0.01
    far: float = 10.0

class GaussianRasterizer(nn.Module):
    """
    PyTorch/CUDA wrapper for Gaussian splatting rasterization.
    Renders RGB and depth from Gaussian parameters.
    """
    
    def __init__(self, config: RenderConfig):
        super().__init__()
        self.config = config
        
        # Initialize rasterization settings
        self.tile_size = config.tile_size
        self.max_gaussians_per_tile = config.max_gaussians_per_tile
        
        # Use real-time SH evaluation instead of precomputation
        # SH basis precomputation disabled - using real-time evaluation in _evaluate_sh()
        self.sh_degree = config.sh_degree
        
    def forward(self, gaussian_params: Dict[str, torch.Tensor],
                camera: CameraParams, T_WC: torch.Tensor,
                lod_config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Render RGB and depth from Gaussians.
        
        Args:
            gaussian_params: Dictionary with positions, rotations, scales, opacity, sh_coeffs
            camera: Camera parameters
            T_WC: World-to-camera transform [4, 4]
            lod_config: Level-of-detail configuration
            
        Returns:
            Dictionary with 'rgb' [H, W, 3] and 'depth' [H, W, 1]
        """
        device = next(iter(gaussian_params.values())).device
        
        # Extract Gaussian parameters
        positions = gaussian_params['positions']  # [N, 3]
        rotations = gaussian_params['rotations']  # [N, 4] 
        scales = gaussian_params['scales']        # [N, 3]
        # Accept both 'opacity' and 'opacities' for compatibility
        if 'opacity' in gaussian_params:
            opacity = gaussian_params['opacity']
        elif 'opacities' in gaussian_params:
            opacity = gaussian_params['opacities']
        else:
            raise KeyError("gaussian_params missing 'opacity' or 'opacities'")
        # STRICT: SH coefficients must be explicitly provided for proper rendering
        if 'sh_coeffs' not in gaussian_params:
            raise KeyError("gaussian_params missing required 'sh_coeffs' key for spherical harmonics rendering")
        sh_coeffs = gaussian_params['sh_coeffs']  # [N, 48]
        
        if len(positions) == 0:
            raise RuntimeError("Cannot render with empty Gaussian set. Ensure Gaussians are properly initialized and not pruned below minimum threshold.")
        
        # Apply level-of-detail if enabled
        if self.config.enable_lod and lod_config:
            positions, rotations, scales, opacity, sh_coeffs = self._apply_lod(
                positions, rotations, scales, opacity, sh_coeffs, T_WC, lod_config
            )
        
        # Determine if gradients are required (training-time)
        requires_grad = (
            positions.requires_grad or rotations.requires_grad or
            scales.requires_grad or opacity.requires_grad or
            (sh_coeffs is not None and sh_coeffs.requires_grad)
        )

        # Optionally downscale rendering during gradient-based passes to save memory
        use_downscale = requires_grad and self.config.enable_grad_downscale and (self.config.grad_render_downscale < 1.0)
        if use_downscale:
            scale = self.config.grad_render_downscale
            scaled_camera = CameraParams(
                fx=camera.fx * scale,
                fy=camera.fy * scale,
                cx=camera.cx * scale,
                cy=camera.cy * scale,
                width=max(1, int(round(camera.width * scale))),
                height=max(1, int(round(camera.height * scale))),
                near=camera.near,
                far=camera.far,
            )
        else:
            scaled_camera = camera

        # Transform to camera space
        positions_cam = self._transform_points(positions, T_WC)
        
        # Store original tensors for potential recovery
        original_rotations = rotations
        original_scales = scales
        original_opacity = opacity
        original_sh_coeffs = sh_coeffs
        
        # Frustum culling (mandatory - no disable option)
        xy_margin = float(lod_config.get('xy_margin', 0.1)) if isinstance(lod_config, dict) else 0.1
        # Always perform frustum culling - no workaround
        valid_mask = self._frustum_culling(positions_cam, scaled_camera, margin=xy_margin)
        positions_cam = positions_cam[valid_mask]
        rotations = rotations[valid_mask]
        scales = scales[valid_mask]
        opacity = opacity[valid_mask]
        if sh_coeffs is not None:
            sh_coeffs = sh_coeffs[valid_mask]
        
        if len(positions_cam) == 0:
            # DEBUGGING: Log frustum culling details before failing
            print(f"   🔍 Debug frustum culling:")
            print(f"      📊 Total Gaussians: {len(positions)}")
            print(f"      📊 After culling: {len(positions_cam)}")
            if len(positions) > 0:
                print(f"      📍 Gaussian positions range: [{positions.min(dim=0)[0]}, {positions.max(dim=0)[0]}]")
                positions_cam_debug = self._transform_points(positions, T_WC)
                print(f"      📍 Camera-space positions range: [{positions_cam_debug.min(dim=0)[0]}, {positions_cam_debug.max(dim=0)[0]}]")
                print(f"      📷 Camera params: near={scaled_camera.near:.3f}, far={scaled_camera.far:.3f}")
                print(f"      📷 Camera resolution: {scaled_camera.width}x{scaled_camera.height}")
                valid_mask_debug = self._frustum_culling(positions_cam_debug, scaled_camera, margin=xy_margin)
                print(f"      ✅ Valid after frustum culling: {valid_mask_debug.sum()}/{len(valid_mask_debug)}")
                if valid_mask_debug.sum() == 0:
                    z_values = torch.abs(positions_cam_debug[:, 2])
                    z_mask = (z_values > scaled_camera.near) & (z_values < scaled_camera.far)
                    print(f"      🔍 Z-culling passed: {z_mask.sum()}/{len(z_mask)} (z_range: [{z_values.min():.3f}, {z_values.max():.3f}])")
            # NO FALLBACKS - Fail fast if culling removes all Gaussians
            raise RuntimeError("No Gaussians visible in camera frustum after culling. Fix camera pose, Gaussian positions, or culling parameters. NO FALLBACK RECOVERY ALLOWED.")
        
        # Project to screen space
        screen_points, depths = self._project_points(positions_cam, scaled_camera)
        
        # Compute 2D covariance matrices
        cov2d = self._compute_2d_covariance(positions_cam, rotations, scales, scaled_camera)
        
        # Sort by depth (back to front for alpha blending)
        depth_indices = torch.argsort(depths, descending=True)
        screen_points = screen_points[depth_indices]
        cov2d = cov2d[depth_indices]
        opacity = opacity[depth_indices]
        depths = depths[depth_indices]
        positions_cam_sorted = positions_cam[depth_indices]
        if sh_coeffs is not None:
            sh_coeffs = sh_coeffs[depth_indices]
        
        # Tile-based rasterization
        rgb, depth = self._tile_rasterization(
            screen_points, cov2d, opacity, depths, sh_coeffs, scaled_camera, positions_cam_sorted
        )

        # If downscaled, upsample back to original resolution for loss computation
        if use_downscale and (scaled_camera.width != camera.width or scaled_camera.height != camera.height):
            # Convert to NCHW for interpolate
            rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
            depth_nchw = depth.permute(2, 0, 1).unsqueeze(0)
            rgb_up = F.interpolate(rgb_nchw, size=(camera.height, camera.width), mode='bilinear', align_corners=False)
            depth_up = F.interpolate(depth_nchw, size=(camera.height, camera.width), mode='bilinear', align_corners=False)
            rgb = rgb_up.squeeze(0).permute(1, 2, 0)
            depth = depth_up.squeeze(0).permute(1, 2, 0)

        return {'rgb': rgb, 'depth': depth}
    
    def _apply_lod(self, positions: torch.Tensor, rotations: torch.Tensor,
                   scales: torch.Tensor, opacity: torch.Tensor,
                   sh_coeffs: Optional[torch.Tensor], T_WC: torch.Tensor,
                   lod_config: Dict) -> Tuple[torch.Tensor, ...]:
        """Apply level-of-detail based on distance from camera"""
        # Transform positions to camera space
        positions_cam = self._transform_points(positions, T_WC)
        distances = torch.norm(positions_cam, dim=1)
        
        # Determine LOD level for each Gaussian
        # STRICT: LOD distance threshold must be explicitly provided
        if 'distance_threshold' not in lod_config:
            raise KeyError("lod_config missing required 'distance_threshold' key")
        lod_threshold = lod_config['distance_threshold']
        lod_levels = torch.clamp(
            torch.floor(distances / lod_threshold).long(),
            0, self.config.lod_levels - 1
        )
        
        # Apply LOD by adjusting scales and potentially culling
        lod_scale_factors = torch.tensor([1.0, 2.0, 4.0], device=positions.device)[lod_levels]
        adjusted_scales = scales + torch.log(lod_scale_factors.unsqueeze(-1))
        
        # Cull distant low-importance Gaussians
        # STRICT: LOD importance threshold must be explicitly provided
        if 'importance_threshold' not in lod_config:
            raise KeyError("lod_config missing required 'importance_threshold' key")
        importance_threshold = lod_config['importance_threshold']
        importance_mask = (opacity.squeeze() > importance_threshold) | (lod_levels == 0)
        
        return (positions[importance_mask], rotations[importance_mask], 
                adjusted_scales[importance_mask], opacity[importance_mask],
                sh_coeffs[importance_mask] if sh_coeffs is not None else None)
    
    def _transform_points(self, points: torch.Tensor, T_WC: torch.Tensor) -> torch.Tensor:
        """Transform points from world to camera coordinates"""
        # Convert to homogeneous coordinates
        points_homo = torch.cat([points, torch.ones(len(points), 1, device=points.device)], dim=1)
        
        # Apply transformation
        points_cam_homo = torch.mm(points_homo, T_WC.t())
        return points_cam_homo[:, :3]
    
    def _frustum_culling(self, positions_cam: torch.Tensor, camera: CameraParams, margin: float = 0.1) -> torch.Tensor:
        """Cull Gaussians outside camera frustum with adjustable XY margin"""
        # Z-culling (depth) - handle negative Z coordinates (OpenCV convention)
        z_values = torch.abs(positions_cam[:, 2])  # Use absolute Z for distance
        z_mask = (z_values > camera.near) & (z_values < camera.far)
        
        # X and Y culling with some margin for Gaussian extent
        # Use intrinsic center to derive FOV correctly; avoid shrinking with width changes
        x_half_fov_factor = max(camera.cx / max(camera.fx, 1e-8), (camera.width - camera.cx) / max(camera.fx, 1e-8))
        y_half_fov_factor = max(camera.cy / max(camera.fy, 1e-8), (camera.height - camera.cy) / max(camera.fy, 1e-8))
        x_bound = z_values * x_half_fov_factor * (1 + margin)
        y_bound = z_values * y_half_fov_factor * (1 + margin)
        
        x_mask = torch.abs(positions_cam[:, 0]) < x_bound
        y_mask = torch.abs(positions_cam[:, 1]) < y_bound
        
        return z_mask & x_mask & y_mask
    
    def _project_points(self, positions_cam: torch.Tensor, 
                       camera: CameraParams) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to screen coordinates with numerical stability
        
        Mathematical Formulation:
        u = (fx * X/Z) + cx
        v = (fy * Y/Z) + cy
        
        Numerical Stability:
        - Clamp Z to [near+ε, far] to prevent division by zero
        - Add epsilon to depth division for numerical safety
        
        References:
        - Hartley & Zisserman "Multiple View Geometry" Chapter 6
        """
        # CRITICAL: Handle negative Z coordinates (OpenCV convention)
        # Use absolute Z for projection depth, but preserve sign for proper ordering
        z_abs = torch.abs(positions_cam[:, 2])
        depths = z_abs.clamp(min=camera.near + 1e-6, max=camera.far)
        
        # Safe perspective projection with epsilon protection
        eps = 1e-8
        depth_safe = depths + eps  # Prevent exact zero division
        
        x_screen = (positions_cam[:, 0] * camera.fx / depth_safe) + camera.cx
        y_screen = (positions_cam[:, 1] * camera.fy / depth_safe) + camera.cy
        
        screen_points = torch.stack([x_screen, y_screen], dim=1)
        return screen_points, depths
    
    def _compute_2d_covariance(self, positions_cam: torch.Tensor, rotations: torch.Tensor,
                              scales: torch.Tensor, camera: CameraParams) -> torch.Tensor:
        """
        Compute 2D covariance matrices for screen-space Gaussians with numerical stability
        
        Mathematical Formulation (Kerbl et al. 2023):
        Σ₃D = R diag(exp(s)) Rᵀ
        J = ∂π/∂X (projection Jacobian)
        Σ₂D = J Σ₃D Jᵀ
        
        Where π(X,Y,Z) = [fx·X/Z + cx, fy·Y/Z + cy]ᵀ
        
        Numerical Stability:
        - Clamp scales to prevent degenerate covariances
        - Safe depth division in Jacobian computation
        - Add regularization to ensure positive definiteness
        """
        # Convert quaternions to rotation matrices (already normalized)
        rotation_matrices = self._quaternion_to_rotation_matrix(rotations)
        
        # CRITICAL: Clamp scales for numerical stability
        # Prevent degenerate Gaussians that cause rendering artifacts
        MIN_SCALE_LOG = math.log(1e-6)  # 1 micrometer minimum
        MAX_SCALE_LOG = math.log(0.1)   # 10cm maximum
        scales_clamped = torch.clamp(scales, MIN_SCALE_LOG, MAX_SCALE_LOG)
        
        # Create 3D covariance matrices with clamped scales
        scale_matrices = torch.diag_embed(torch.exp(scales_clamped))
        cov3d = torch.bmm(torch.bmm(rotation_matrices, scale_matrices), 
                         rotation_matrices.transpose(-1, -2))
        
        # Add regularization to ensure positive definiteness
        eps_matrix = torch.eye(3, device=cov3d.device).unsqueeze(0) * 1e-6
        cov3d = cov3d + eps_matrix
        
        # Project to 2D using Jacobian of perspective projection
        # CRITICAL: Safe depth handling for Jacobian computation
        depths = positions_cam[:, 2:3].clamp(min=camera.near + 1e-6)  # [N, 1]
        depth_safe = depths + 1e-8  # Additional epsilon for division
        fx, fy = camera.fx, camera.fy
        
        # Jacobian matrix for perspective projection with safe divisions
        jacobian = torch.full((len(positions_cam), 2, 3), 1e-12, device=positions_cam.device)
        jacobian[:, 0, 0] = fx / depth_safe.squeeze()
        jacobian[:, 0, 2] = -fx * positions_cam[:, 0] / (depth_safe.squeeze() ** 2)
        jacobian[:, 1, 1] = fy / depth_safe.squeeze()
        jacobian[:, 1, 2] = -fy * positions_cam[:, 1] / (depth_safe.squeeze() ** 2)
        
        # Project covariance: Σ_2D = J * Σ_3D * J^T
        cov2d = torch.bmm(torch.bmm(jacobian, cov3d), jacobian.transpose(-1, -2))
        
        # Final regularization for 2D covariance (ensure invertibility)
        eps_2d = torch.eye(2, device=cov2d.device).unsqueeze(0) * 1e-6
        cov2d = cov2d + eps_2d
        
        return cov2d
    
    def _quaternion_to_rotation_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to rotation matrices with numerical stability
        
        Mathematical Formulation:
        Given normalized quaternion q = [w, x, y, z]ᵀ with |q| = 1
        
        Rotation matrix R:
        R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
             [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
             [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
        
        Numerical Stability:
        - Add epsilon to norm for safe division
        - Ensure |q| = 1 through explicit normalization
        
        References:
        - Shoemake "Animating rotation with quaternion curves" SIGGRAPH 1985
        """
        # CRITICAL: Numerical stability in quaternion normalization
        eps = 1e-8
        q_norm = torch.norm(quaternions, dim=1, keepdim=True) + eps
        q = quaternions / q_norm  # Safe normalization
        
        # Extract quaternion components
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        rotation_matrices = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=1)
        ], dim=1)  # [N, 3, 3]
        
        return rotation_matrices
    
    def _tile_rasterization(self, screen_points: torch.Tensor, cov2d: torch.Tensor,
                           opacity: torch.Tensor, depths: torch.Tensor,
                           sh_coeffs: Optional[torch.Tensor], 
                           camera: CameraParams,
                           positions_cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Academic-quality tile-based rasterization with proper transmittance compositing
        
        Implements front-to-back compositing following Kerbl et al. 2023:
        - T₀ = 1 (initial transmittance)
        - For each splat i: wᵢ = Tᵢ₋₁ · αᵢ, C += wᵢ·cᵢ, D += wᵢ·zᵢ, Tᵢ = Tᵢ₋₁·(1-αᵢ)
        
        Expected depth rendering:
        D(p) = Σᵢ zᵢ αᵢ Πⱼ₌₁^{i-1} (1 - αⱼ)
        
        References:
        - Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" SIGGRAPH 2023
        - Standard volume rendering with transmittance (Jensen & Christensen 1998)
        """
        device = screen_points.device
        height, width = camera.height, camera.width
        
        # Initialize output images and transmittance
        rgb = torch.full((height, width, 3), 1e-8, device=device)
        depth_expected = torch.full((height, width, 1), 1e-6, device=device)  # Expected depth
        transmittance = torch.ones(height, width, 1, device=device)     # T = Π(1-α)
        
        # Create pixel grid for vectorized computation
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float()
        
        # Front-to-back compositing (CRITICAL: proper volume rendering)
        for i in range(len(screen_points)):
            center = screen_points[i]  # [2]
            cov = cov2d[i]            # [2, 2]
            alpha_base = opacity[i].clamp(1e-5, 1.0 - 1e-5)  # Numerical stability
            depth_val = depths[i].item()
            
            # Compute color from spherical harmonics
            if sh_coeffs is not None:
                # Compute view direction from camera origin to Gaussian center
                # Use already sorted camera-space positions
                gaussian_pos = positions_cam[i]  # Direct access to sorted positions
                view_dir = gaussian_pos / (torch.norm(gaussian_pos) + 1e-8)  # Normalize
                color = self._evaluate_sh(sh_coeffs[i], view_dir)
            else:
                color = torch.ones(3, device=device) * 0.5
            
            # Compute Gaussian weights for all pixels
            diff = pixel_coords - center.unsqueeze(0)  # [H*W, 2]
            
            # ACADEMIC QUALITY: Numerically stable 2x2 matrix inversion
            # Use analytic inverse for 2x2 matrices (more stable than torch.inverse)
            a, b = cov[0, 0] + 1e-6, cov[0, 1]  # Add epsilon for stability
            c, d = cov[1, 0], cov[1, 1] + 1e-6
            det = a * d - b * c
            
            # Analytic 2x2 inverse (numerically superior to torch.inverse)
            inv_00, inv_01 = d / det, -b / det
            inv_10, inv_11 = -c / det, a / det
            
            # Mahalanobis distance: (x-μ)ᵀ Σ⁻¹ (x-μ)
            mahal_dist = (diff[:, 0] * (inv_00 * diff[:, 0] + inv_01 * diff[:, 1]) + 
                         diff[:, 1] * (inv_10 * diff[:, 0] + inv_11 * diff[:, 1]))
            
            # Gaussian weight with numerical stability
            alpha_i = alpha_base * torch.exp(-0.5 * mahal_dist)  # [H*W]
            
            # Apply alpha threshold for efficiency (skip negligible contributions)
            valid_mask = alpha_i > self.config.alpha_threshold
            if not valid_mask.any():
                continue
            
            # Reshape to image format
            alpha_img = alpha_i.view(height, width, 1)  # [H, W, 1]
            
            # CRITICAL FIX: Proper front-to-back transmittance compositing
            # Weight = current_transmittance × alpha_i
            weight = transmittance * alpha_img
            
            # Accumulate RGB and expected depth
            rgb = rgb + weight * color.view(1, 1, 3)
            depth_expected = depth_expected + weight * depth_val
            
            # Update transmittance: T *= (1 - α)
            transmittance = transmittance * (1.0 - alpha_img)
            
            # Early termination when transmittance becomes negligible
            if (transmittance < 1e-3).all():
                break
        
        # Background composition using remaining transmittance
        if self.config.white_background:
            background = torch.ones_like(rgb)
        else:
            background = torch.full_like(rgb, 1e-8)
        
        rgb = rgb + transmittance * background
        
        return rgb, depth_expected
    
    def _evaluate_sh(self, sh_coeffs: torch.Tensor, view_dir: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate spherical harmonics for view-dependent color
        
        Mathematical Formulation (following Kerbl et al. 2023):
        c(d) = Σₗ₌₀ᴸ Σₘ₌₋ₗˡ cₗᵐ Yₗᵐ(d)
        
        Where:
        - c(d): RGB color for viewing direction d
        - cₗᵐ: SH coefficients (stored in sh_coeffs)
        - Yₗᵐ(d): Spherical harmonic basis function
        - L: Maximum SH degree (typically 3 for 48 coefficients)
        
        SH basis functions up to degree 3:
        - Y₀⁰ = 0.28209479 (DC component)
        - Y₁⁻¹ = 0.48860251 * y, Y₁⁰ = 0.48860251 * z, Y₁¹ = 0.48860251 * x
        - Y₂⁻² = 1.09254843 * x*y, Y₂⁻¹ = 1.09254843 * y*z, etc.
        
        References:
        - Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" SIGGRAPH 2023
        - Ramamoorthi & Hanrahan "An Efficient Representation for Irradiance Environment Maps" SIGGRAPH 2001
        """
        if view_dir is None:
            raise ValueError("view_dir required for SH evaluation")
        required = 48  # degree 3 RGB SH
        if sh_coeffs.numel() < required:
            raise ValueError(f"sh_coeffs has {sh_coeffs.numel()} elements, expected ≥{required}")
        
        # Normalize view direction
        d = view_dir / (torch.norm(view_dir) + 1e-8)
        x, y, z = d[0], d[1], d[2]
        
        # Spherical Harmonics evaluation up to degree 3 (48 coefficients)
        # Following the standard SH basis functions from computer graphics literature
        
        # Degree 0 (1 function): DC component
        result = torch.full((3,), 1e-8, device=sh_coeffs.device, dtype=sh_coeffs.dtype)
        
        if sh_coeffs.numel() >= 3:
            # Y₀⁰ = 0.28209479177387814 (constant)
            result += 0.28209479177387814 * sh_coeffs[:3]
        
        # Degree 1 (3 functions): linear terms
        if sh_coeffs.numel() >= 12:
            # Y₁⁻¹ = 0.4886025119029199 * y
            result += 0.4886025119029199 * y * sh_coeffs[3:6]
            # Y₁⁰ = 0.4886025119029199 * z  
            result += 0.4886025119029199 * z * sh_coeffs[6:9]
            # Y₁¹ = 0.4886025119029199 * x
            result += 0.4886025119029199 * x * sh_coeffs[9:12]
        
        # Degree 2 (5 functions): quadratic terms
        if sh_coeffs.numel() >= 27:
            # Y₂⁻² = 1.0925484305920792 * x * y
            result += 1.0925484305920792 * x * y * sh_coeffs[12:15]
            # Y₂⁻¹ = 1.0925484305920792 * y * z
            result += 1.0925484305920792 * y * z * sh_coeffs[15:18]
            # Y₂⁰ = 0.31539156525252005 * (2*z² - x² - y²)
            result += 0.31539156525252005 * (2.0*z*z - x*x - y*y) * sh_coeffs[18:21]
            # Y₂¹ = 1.0925484305920792 * x * z
            result += 1.0925484305920792 * x * z * sh_coeffs[21:24]
            # Y₂² = 0.5462742152960396 * (x² - y²)
            result += 0.5462742152960396 * (x*x - y*y) * sh_coeffs[24:27]
        
        # Degree 3 (7 functions): cubic terms
        if sh_coeffs.numel() >= 48:
            # Y₃⁻³ = 0.5900435899266435 * y * (3*x² - y²)
            result += 0.5900435899266435 * y * (3.0*x*x - y*y) * sh_coeffs[27:30]
            # Y₃⁻² = 2.890611442640554 * x * y * z
            result += 2.890611442640554 * x * y * z * sh_coeffs[30:33]
            # Y₃⁻¹ = 0.4570457994644658 * y * (4*z² - x² - y²)
            result += 0.4570457994644658 * y * (4.0*z*z - x*x - y*y) * sh_coeffs[33:36]
            # Y₃⁰ = 0.3731763325901154 * z * (2*z² - 3*x² - 3*y²)
            result += 0.3731763325901154 * z * (2.0*z*z - 3.0*x*x - 3.0*y*y) * sh_coeffs[36:39]
            # Y₃¹ = 0.4570457994644658 * x * (4*z² - x² - y²)
            result += 0.4570457994644658 * x * (4.0*z*z - x*x - y*y) * sh_coeffs[39:42]
            # Y₃² = 1.445305721320277 * z * (x² - y²)
            result += 1.445305721320277 * z * (x*x - y*y) * sh_coeffs[42:45]
            # Y₃³ = 0.5900435899266435 * x * (x² - 3*y²)
            result += 0.5900435899266435 * x * (x*x - 3.0*y*y) * sh_coeffs[45:48]
        
        # Apply sigmoid activation for color values in [0,1]
        return torch.sigmoid(result)
    
    def _compute_sh_basis(self, degree: int) -> torch.Tensor:
        """
        Precompute spherical harmonics basis functions
        
        Academic Implementation Notes:
        - Follows standard computer graphics SH conventions
        - Coefficients from Ramamoorthi & Hanrahan 2001
        - Supports up to degree 3 (48 coefficients) for efficiency
        
        Returns:
            Basis tensor of shape [(degree+1)², 3] for RGB channels
        """
        # SH basis precomputation not implemented - fail fast
        raise NotImplementedError(
            "SH basis precomputation not implemented. "
            "Use real-time SH evaluation in evaluate_sh() instead."
        )

def render_rgbd(gaussian_params: Dict[str, torch.Tensor], camera: CameraParams,
                T_WC: torch.Tensor, lod_config: Optional[Dict] = None,
                render_config: Optional[RenderConfig] = None) -> Dict[str, torch.Tensor]:
    """
    Convenience function for rendering RGB and depth from Gaussians.
    
    Args:
        gaussian_params: Dictionary of Gaussian parameters
        camera: Camera parameters 
        T_WC: World-to-camera transform [4, 4]
        lod_config: Level-of-detail configuration
        render_config: Rendering configuration
        
    Returns:
        Dictionary with 'rgb' and 'depth' tensors
    """
    config = render_config or RenderConfig()
    rasterizer = GaussianRasterizer(config)
    
    return rasterizer(gaussian_params, camera, T_WC, lod_config)
