# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Jacobian diagnostics for GaussianFeels optimization.
Provides validation utilities for Theseus optimization with Gaussian splatting.
"""

import time
import copy
from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import theseus as th
from termcolor import cprint


class JacobianTimer:
    """GPU timing utilities for jacobian validation"""
    
    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.start_mem = 0.0
        
    def start_timer(self):
        """Start GPU timing"""
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.max_memory_allocated() / 1048576
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop_timer(self) -> Tuple[float, float]:
        """Stop GPU timing and return (time_ms, memory_mb)"""
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            end_mem = torch.cuda.max_memory_allocated() / 1048576
            forward_mem = end_mem - self.start_mem
            forward_time = self.start_event.elapsed_time(self.end_event)
            return forward_time, forward_mem
        else:
            elapsed = (time.perf_counter() - self.start_time) * 1000
            return elapsed, 0.0


@torch.no_grad()
def check_gaussian_jacobians(cf: th.CostFunction, 
                           tolerance: float = 1.0e-3,
                           verbose: bool = True) -> bool:
    """
    Check jacobians for Gaussian splatting cost functions.
    Jacobian validation for Gaussian field optimization.
    
    Args:
        cf: Theseus cost function to validate
        tolerance: Tolerance for jacobian comparison
        verbose: Print detailed output
        
    Returns:
        True if jacobians pass validation, False otherwise
    """
    from theseus.core.cost_function import _tmp_tensors

    optim_vars: List[th.Manifold] = list(cf.optim_vars)
    aux_vars = list(cf.aux_vars)

    def autograd_fn(*optim_var_tensors):
        for v, t in zip(optim_vars, optim_var_tensors):
            v.update(t)
        return cf.error()

    jacobians_valid = True
    
    with _tmp_tensors(optim_vars), _tmp_tensors(aux_vars):
        timer = JacobianTimer()
        timer.start_timer()
        
        # Compute jacobians using autograd
        autograd_jac = torch.autograd.functional.jacobian(
            autograd_fn, tuple(v.tensor for v in optim_vars)
        )
        
        # Compute analytical jacobians
        jac, _ = cf.jacobians()
        
        forward_time, forward_mem = timer.stop_timer()
        
        if verbose:
            print(f"\nJacobian validation for {cf.name}")
            print(f"Computation time: {forward_time:.2f}ms, Memory: {forward_mem:.1f}MB")
        
        for idx, v in enumerate(optim_vars):
            j1 = jac[idx]  # Analytical jacobian
            j2 = autograd_jac[idx]  # Autograd jacobian
            
            # Handle batch dimension differences
            j2_sparse = j2[:, :, 0, :]
            j2_sparse_manifold = v.project(j2_sparse, is_sparse=True)
            
            max_diff = (j1 - j2_sparse_manifold).abs().max()
            
            if verbose:
                print(f"Variable '{v.name}': Jacobian shapes {j1.shape} vs {j2_sparse_manifold.shape}")
                print(f"Max difference: {max_diff:.6f} (tolerance: {tolerance})")
            
            if max_diff > tolerance:
                if verbose:
                    cprint(f"Jacobian for variable '{v.name}' FAILED validation", "red")
                jacobians_valid = False
            else:
                if verbose:
                    cprint(f"Jacobian for variable '{v.name}' passed validation", "green")
    
    return jacobians_valid


class GaussianFieldCostFunction(th.CostFunction):
    """
    Cost function for Gaussian field optimization with analytical jacobians.
    TSDF-like cost function for Gaussian splatting.
    """
    
    def __init__(self,
                 gaussian_params: th.Vector,
                 target_points: torch.Tensor,
                 cost_weight: th.CostWeight,
                 field_evaluator: Callable,
                 name: Optional[str] = "gaussian_field_loss"):
        super().__init__(cost_weight, name=name)
        
        if not isinstance(gaussian_params, (th.Vector, th.SE3)):
            raise ValueError(f"Gaussian params must be th.Vector or th.SE3, got {type(gaussian_params)}")
            
        self.gaussian_params = gaussian_params
        self.target_points = target_points
        self.field_evaluator = field_evaluator
        self._dim = target_points.shape[0]
        
        self.register_optim_vars(["gaussian_params"])

    def error(self) -> torch.Tensor:
        """Compute Gaussian field reconstruction error"""
        predicted_values = self.field_evaluator(self.target_points, self.gaussian_params)
        return predicted_values  # Error relative to target (e.g., depth, SDF)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compute analytical jacobians w.r.t. Gaussian parameters"""
        predicted_values = self.field_evaluator(
            self.target_points, self.gaussian_params, return_jacobian=True
        )
        
        if isinstance(predicted_values, tuple):
            values, jac = predicted_values
            return [jac], values
        else:
            # STRICT: Analytical jacobian required - no numerical fallback
            raise ValueError("Cost function must return (values, jacobian) tuple for jacobian validation - no numerical fallback allowed")

    def _compute_numerical_jacobian(self, eps: float = 1e-6) -> torch.Tensor:
        """DEPRECATED: Numerical jacobian computation - should not be used in strict mode"""
        param_tensor = self.gaussian_params.tensor.clone()
        jacobian = torch.zeros(self._dim, param_tensor.numel(), device=param_tensor.device)
        
        base_error = self.error()
        
        for i in range(param_tensor.numel()):
            # Forward difference
            param_tensor.view(-1)[i] += eps
            self.gaussian_params.update(param_tensor)
            
            error_plus = self.field_evaluator(self.target_points, self.gaussian_params)
            jacobian[:, i] = (error_plus - base_error) / eps
            
            # Restore parameter
            param_tensor.view(-1)[i] -= eps
            self.gaussian_params.update(param_tensor)
        
        return jacobian.unsqueeze(0)  # Add batch dimension

    def dim(self) -> int:
        return self._dim

    def _copy_impl(self, new_name: Optional[str] = None) -> "GaussianFieldCostFunction":
        return GaussianFieldCostFunction(
            self.gaussian_params.copy(),
            self.target_points.clone(),
            self.weight.copy(),
            self.field_evaluator,
            name=new_name
        )


class GaussianSplattingDiagnostics:
    """
    Comprehensive jacobian diagnostics for Gaussian splatting optimization.
    Provides validation for different components: transforms, field evaluation, losses.
    """
    
    def __init__(self, device: torch.device = None):
        if device is None:
            raise ValueError("device parameter is required - no fallback device selection allowed")
        self.device = device
        self.timer = JacobianTimer()
        
    def validate_transform_jacobians(self, 
                                   poses: th.SE3,
                                   points: torch.Tensor,
                                   transform_fn: Callable) -> bool:
        """Validate jacobians for SE(3) pose transforms"""
        
        class TransformCostFunction(th.CostFunction):
            def __init__(self, pose, points, transform_fn, cost_weight):
                super().__init__(cost_weight)
                self.pose = pose
                self.points = points
                self.transform_fn = transform_fn
                self.register_optim_vars(["pose"])

            def error(self) -> torch.Tensor:
                return self.transform_fn(self.points, self.pose.to_matrix())

            def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
                transformed_points, jac = self.transform_fn(
                    self.points, self.pose.to_matrix(), return_jacobian=True
                )
                return [jac], transformed_points

            def dim(self) -> int:
                return self.points.shape[0] * 3

            def _copy_impl(self, new_name=None):
                return TransformCostFunction(
                    self.pose.copy(), self.points, self.transform_fn, self.weight.copy()
                )

        cost_weight = th.ScaleCostWeight(1.0)
        cf = TransformCostFunction(poses, points, transform_fn, cost_weight)
        
        return check_gaussian_jacobians(cf, verbose=True)
        
    def validate_gaussian_field_jacobians(self,
                                        gaussian_params: th.Vector,
                                        sample_points: torch.Tensor,
                                        field_evaluator: Callable) -> bool:
        """Validate jacobians for Gaussian field evaluation"""
        
        cost_weight = th.ScaleCostWeight(1.0)
        cf = GaussianFieldCostFunction(
            gaussian_params, sample_points, cost_weight, field_evaluator
        )
        
        return check_gaussian_jacobians(cf, verbose=True)
    
    def profile_optimization_step(self,
                                objective: th.Objective,
                                optimizer_cls: type,
                                max_iterations: int = 10) -> Dict[str, Any]:
        """Profile a complete optimization step with timing"""
        
        self.timer.start_timer()
        
        # Create optimizer
        optimizer = optimizer_cls(objective, max_iterations=max_iterations)
        
        # Create Theseus layer
        theseus_optim = th.TheseusLayer(optimizer)
        
        # Time the forward pass
        with torch.no_grad():
            theseus_inputs = {var.name: var for var in objective.optim_vars}
            updated_inputs, info = theseus_optim.forward(theseus_inputs)
            
        forward_time, forward_mem = self.timer.stop_timer()
        
        # Collect performance metrics
        metrics = {
            'forward_time_ms': forward_time,
            'memory_usage_mb': forward_mem,
            'converged': info.status[0] != th.NonlinearOptimizerStatus.FAIL,
            'final_error': info.last_err[-1].item() if info.last_err else float('inf'),
            'iterations': len(info.last_err) if info.last_err else 0,
            'optimizer_class': optimizer_cls.__name__
        }
        
        return metrics

    def run_comprehensive_validation(self,
                                   poses: th.SE3,
                                   gaussian_params: th.Vector,
                                   sample_points: torch.Tensor,
                                   transform_fn: Callable,
                                   field_evaluator: Callable) -> Dict[str, bool]:
        """Run comprehensive jacobian validation for all components"""
        
        results = {}
        
        print("Running comprehensive Gaussian splatting jacobian validation...")
        print("=" * 70)
        
        # Test transform jacobians
        print("\n1. Validating SE(3) transform jacobians...")
        results['transforms'] = self.validate_transform_jacobians(
            poses, sample_points, transform_fn
        )
        
        # Test field evaluation jacobians  
        print("\n2. Validating Gaussian field evaluation jacobians...")
        results['field_evaluation'] = self.validate_gaussian_field_jacobians(
            gaussian_params, sample_points, field_evaluator  
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY:")
        all_passed = all(results.values())
        status_color = "green" if all_passed else "red" 
        for component, passed in results.items():
            print(f"- {component}: {'PASSED' if passed else 'FAILED'}")
            
        cprint(f"\nOverall validation: {'PASSED' if all_passed else 'FAILED'}", status_color)
        
        return results


def create_test_gaussian_field_evaluator(n_gaussians: int = 1000) -> Callable:
    """Create a test Gaussian field evaluator for jacobian validation"""
    
    def gaussian_field_eval(points: torch.Tensor, 
                          params: th.Vector,
                          return_jacobian: bool = False) -> torch.Tensor:
        """
        Simplified Gaussian field evaluation for testing jacobians.
        In practice, this would be your actual Gaussian splatting evaluation.
        """
        # Mock Gaussian field evaluation
        param_tensor = params.tensor.view(-1, 7)  # [xyz, scale, rotation]
        
        # Simple distance-based evaluation for testing
        centers = param_tensor[:, :3]  # Gaussian centers
        
        # Compute distances from points to Gaussian centers
        distances = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
        values = torch.exp(-distances.min(dim=1)[0])  # Closest Gaussian contribution
        
        if return_jacobian:
            # Simplified jacobian computation
            jac = torch.zeros(values.shape[0], param_tensor.numel(), device=points.device)
            # In practice, compute analytical jacobians here
            return values, jac.unsqueeze(0)
        
        return values
    
    return gaussian_field_eval


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Gaussian splatting jacobian diagnostics...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diagnostics = GaussianSplattingDiagnostics(device)
    
    # Create test data
    n_points = 1000
    n_gaussians = 100
    
    sample_points = torch.randn(n_points, 3, device=device)
    poses = th.SE3(tensor=torch.eye(4, device=device).unsqueeze(0)[:, :3, :])
    gaussian_params = th.Vector(tensor=torch.randn(n_gaussians * 7, device=device))
    
    # Create test evaluators
    def test_transform_fn(points, pose_matrix, return_jacobian=False):
        transformed = torch.matmul(pose_matrix[:3, :3], points.T).T + pose_matrix[:3, 3]
        if return_jacobian:
            jac = torch.zeros(points.shape[0] * 3, 6, device=points.device)
            return transformed, jac.unsqueeze(0)
        return transformed
    
    field_evaluator = create_test_gaussian_field_evaluator(n_gaussians)
    
    # Run validation
    results = diagnostics.run_comprehensive_validation(
        poses, gaussian_params, sample_points, test_transform_fn, field_evaluator
    )
    
    print(f"\nValidation results: {results}")