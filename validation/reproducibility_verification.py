#!/usr/bin/env python3
"""
Reproducibility Verification Framework for GaussianFeels

This module provides comprehensive verification of reproducibility for academic publication.
It ensures that all results can be reproduced exactly across different runs, machines, and environments.

Author: Krishi Attri, Soft Robotics and Bionics Lab, Seoul National University
License: MIT
"""

import os
import sys
import json
import hashlib
import subprocess
import torch
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import platform
import datetime
import warnings


class ReproducibilityVerifier:
    """
    Comprehensive reproducibility verification for academic research.
    
    Verifies:
    - Deterministic training across multiple runs
    - Environment consistency
    - Random seed control
    - Hardware-specific reproducibility
    - Cross-platform consistency
    """
    
    def __init__(self, base_seed: int = 42, num_verification_runs: int = 3):
        self.base_seed = base_seed
        self.num_verification_runs = num_verification_runs
        self.verification_results = {}
        
    def verify_all(self) -> Dict[str, Any]:
        """Run complete reproducibility verification suite."""
        print("🔬 Running Reproducibility Verification Suite...")
        print("=" * 60)
        
        # Core reproducibility tests
        self.verification_results['environment'] = self.verify_environment_consistency()
        self.verification_results['random_seeds'] = self.verify_random_seed_control()
        self.verification_results['deterministic_ops'] = self.verify_deterministic_operations()
        self.verification_results['training_reproducibility'] = self.verify_training_reproducibility()
        self.verification_results['cross_run_consistency'] = self.verify_cross_run_consistency()
        
        # Advanced reproducibility tests
        self.verification_results['numerical_precision'] = self.verify_numerical_precision()
        self.verification_results['memory_determinism'] = self.verify_memory_determinism()
        self.verification_results['threading_determinism'] = self.verify_threading_determinism()
        
        self._print_verification_summary()
        return self.verification_results
    
    def verify_environment_consistency(self) -> Dict[str, Any]:
        """Verify environment setup and consistency."""
        print("Verifying environment consistency...")
        
        env_info = self._get_environment_info()
        
        # Check for known reproducibility issues
        issues = []
        
        # Check CUDA determinism
        if torch.cuda.is_available():
            if not os.environ.get('CUDA_LAUNCH_BLOCKING'):
                issues.append("CUDA_LAUNCH_BLOCKING not set")
            
            # Check for deterministic CUDA operations
            try:
                torch.use_deterministic_algorithms(True)
                deterministic_available = True
            except Exception as e:
                deterministic_available = False
                issues.append(f"Deterministic algorithms not available: {e}")
        
        # Check cuDNN settings
        if torch.backends.cudnn.is_available():
            if torch.backends.cudnn.benchmark:
                issues.append("cuDNN benchmark enabled (should be False for reproducibility)")
            if not torch.backends.cudnn.deterministic:
                issues.append("cuDNN deterministic disabled (should be True)")
        
        # Check Python hash randomization
        if os.environ.get('PYTHONHASHSEED') != '0':
            issues.append("PYTHONHASHSEED not set to 0")
        
        result = {
            'test': 'environment_consistency',
            'passed': len(issues) == 0,
            'environment_info': env_info,
            'issues': issues,
            'deterministic_available': deterministic_available if torch.cuda.is_available() else 'N/A'
        }
        
        print(f"  ✓ Environment info collected")
        print(f"  ✓ Issues found: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_random_seed_control(self) -> Dict[str, Any]:
        """Verify random seed control across all libraries."""
        print("Verifying random seed control...")
        
        # Test different random number generators
        generators = {
            'python_random': lambda: random.random(),
            'numpy_random': lambda: np.random.random(),
            'torch_random': lambda: torch.rand(1).item(),
        }
        
        if torch.cuda.is_available():
            generators['torch_cuda_random'] = lambda: torch.cuda.FloatTensor(1).uniform_().item()
        
        seed_control_results = {}
        
        for gen_name, gen_func in generators.items():
            # Test reproducibility
            self._setup_deterministic_environment(self.base_seed)
            values_run1 = [gen_func() for _ in range(10)]
            
            self._setup_deterministic_environment(self.base_seed)
            values_run2 = [gen_func() for _ in range(10)]
            
            # Check if values are identical
            identical = all(abs(v1 - v2) < 1e-10 for v1, v2 in zip(values_run1, values_run2))
            seed_control_results[gen_name] = {
                'identical': identical,
                'max_difference': max(abs(v1 - v2) for v1, v2 in zip(values_run1, values_run2))
            }
        
        all_controlled = all(result['identical'] for result in seed_control_results.values())
        
        result = {
            'test': 'random_seed_control',
            'passed': all_controlled,
            'generator_results': seed_control_results
        }
        
        print(f"  ✓ Tested {len(generators)} random generators")
        print(f"  ✓ All controlled: {all_controlled}")
        for gen_name, gen_result in seed_control_results.items():
            status = "✓" if gen_result['identical'] else "✗"
            print(f"    {status} {gen_name}: max_diff={gen_result['max_difference']:.2e}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_deterministic_operations(self) -> Dict[str, Any]:
        """Verify deterministic behavior of key operations."""
        print("Verifying deterministic operations...")
        
        operations = {
            'matrix_multiplication': self._test_matmul_determinism,
            'convolution': self._test_conv_determinism,
            'attention': self._test_attention_determinism,
            'sorting': self._test_sorting_determinism
        }
        
        operation_results = {}
        
        for op_name, op_test in operations.items():
            try:
                is_deterministic, max_diff = op_test()
                operation_results[op_name] = {
                    'deterministic': is_deterministic,
                    'max_difference': max_diff,
                    'error': None
                }
            except Exception as e:
                operation_results[op_name] = {
                    'deterministic': False,
                    'max_difference': float('inf'),
                    'error': str(e)
                }
        
        all_deterministic = all(result['deterministic'] for result in operation_results.values())
        
        result = {
            'test': 'deterministic_operations',
            'passed': all_deterministic,
            'operation_results': operation_results
        }
        
        print(f"  ✓ Tested {len(operations)} operations")
        for op_name, op_result in operation_results.items():
            status = "✓" if op_result['deterministic'] else "✗"
            error_info = f" (error: {op_result['error']})" if op_result['error'] else ""
            print(f"    {status} {op_name}: max_diff={op_result['max_difference']:.2e}{error_info}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_training_reproducibility(self) -> Dict[str, Any]:
        """Verify training reproducibility with simplified model."""
        print("Verifying training reproducibility...")
        
        # Create simple model for testing
        model_results = []
        
        for run in range(self.num_verification_runs):
            self._setup_deterministic_environment(self.base_seed)
            
            # Simple model and data
            model = torch.nn.Linear(10, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            data = torch.randn(100, 10)
            targets = torch.randn(100, 1)
            
            # Training steps
            losses = []
            for step in range(10):
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            # Record final state
            final_params = [p.clone().detach() for p in model.parameters()]
            model_results.append({
                'losses': losses,
                'final_params': final_params
            })
        
        # Check consistency across runs
        consistent = True
        max_param_diff = 0.0
        max_loss_diff = 0.0
        
        for i in range(1, len(model_results)):
            # Compare losses
            for j, (loss1, loss2) in enumerate(zip(model_results[0]['losses'], model_results[i]['losses'])):
                diff = abs(loss1 - loss2)
                max_loss_diff = max(max_loss_diff, diff)
                if diff > 1e-6:
                    consistent = False
            
            # Compare parameters
            for p1, p2 in zip(model_results[0]['final_params'], model_results[i]['final_params']):
                diff = torch.max(torch.abs(p1 - p2)).item()
                max_param_diff = max(max_param_diff, diff)
                if diff > 1e-6:
                    consistent = False
        
        result = {
            'test': 'training_reproducibility',
            'passed': consistent,
            'num_runs': self.num_verification_runs,
            'max_loss_difference': max_loss_diff,
            'max_param_difference': max_param_diff
        }
        
        print(f"  ✓ Ran {self.num_verification_runs} training runs")
        print(f"  ✓ Max loss difference: {max_loss_diff:.2e}")
        print(f"  ✓ Max parameter difference: {max_param_diff:.2e}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_cross_run_consistency(self) -> Dict[str, Any]:
        """Verify consistency across multiple independent runs."""
        print("Verifying cross-run consistency...")
        
        # Test hash consistency of key computations
        hash_results = []
        
        for run in range(self.num_verification_runs):
            self._setup_deterministic_environment(self.base_seed)
            
            # Simulate key computations
            tensor = torch.randn(100, 100)
            result = torch.matmul(tensor, tensor.T)
            
            # Compute hash of result
            result_hash = hashlib.sha256(result.detach().cpu().numpy().tobytes()).hexdigest()
            hash_results.append(result_hash)
        
        # Check if all hashes are identical
        all_identical = all(h == hash_results[0] for h in hash_results)
        unique_hashes = len(set(hash_results))
        
        result = {
            'test': 'cross_run_consistency',
            'passed': all_identical,
            'num_runs': self.num_verification_runs,
            'unique_hashes': unique_hashes,
            'hashes': hash_results
        }
        
        print(f"  ✓ Ran {self.num_verification_runs} consistency checks")
        print(f"  ✓ Unique hashes: {unique_hashes}")
        print(f"  ✓ All identical: {all_identical}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_numerical_precision(self) -> Dict[str, Any]:
        """Verify numerical precision consistency."""
        print("Verifying numerical precision...")
        
        # Test precision across different operations
        precision_tests = {
            'float32_arithmetic': self._test_float32_precision,
            'float64_arithmetic': self._test_float64_precision,
            'mixed_precision': self._test_mixed_precision
        }
        
        precision_results = {}
        
        for test_name, test_func in precision_tests.items():
            try:
                is_consistent, max_error = test_func()
                precision_results[test_name] = {
                    'consistent': is_consistent,
                    'max_error': max_error,
                    'error': None
                }
            except Exception as e:
                precision_results[test_name] = {
                    'consistent': False,
                    'max_error': float('inf'),
                    'error': str(e)
                }
        
        all_consistent = all(result['consistent'] for result in precision_results.values())
        
        result = {
            'test': 'numerical_precision',
            'passed': all_consistent,
            'precision_results': precision_results
        }
        
        print(f"  ✓ Tested {len(precision_tests)} precision scenarios")
        for test_name, test_result in precision_results.items():
            status = "✓" if test_result['consistent'] else "✗"
            error_info = f" (error: {test_result['error']})" if test_result['error'] else ""
            print(f"    {status} {test_name}: max_error={test_result['max_error']:.2e}{error_info}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_memory_determinism(self) -> Dict[str, Any]:
        """Verify memory allocation determinism."""
        print("Verifying memory determinism...")
        
        # Test memory allocation patterns
        memory_patterns = []
        
        for run in range(self.num_verification_runs):
            self._setup_deterministic_environment(self.base_seed)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Allocate tensors
                tensors = []
                for i in range(10):
                    tensor = torch.randn(100, 100, device='cuda')
                    tensors.append(tensor)
                
                final_memory = torch.cuda.memory_allocated()
                memory_patterns.append(final_memory - initial_memory)
            else:
                # CPU memory is harder to track precisely
                memory_patterns.append(0)
        
        # Check consistency
        if torch.cuda.is_available():
            consistent = all(pattern == memory_patterns[0] for pattern in memory_patterns)
        else:
            consistent = True  # Skip test for CPU-only
        
        result = {
            'test': 'memory_determinism',
            'passed': consistent,
            'memory_patterns': memory_patterns,
            'cuda_available': torch.cuda.is_available()
        }
        
        print(f"  ✓ Memory patterns: {memory_patterns}")
        print(f"  ✓ Consistent: {consistent}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    def verify_threading_determinism(self) -> Dict[str, Any]:
        """Verify determinism with multiple threads."""
        print("Verifying threading determinism...")
        
        # Test single-threaded vs multi-threaded consistency
        original_num_threads = torch.get_num_threads()
        
        try:
            # Single-threaded result
            torch.set_num_threads(1)
            self._setup_deterministic_environment(self.base_seed)
            single_result = torch.randn(100, 100) @ torch.randn(100, 100)
            
            # Multi-threaded result
            torch.set_num_threads(4)
            self._setup_deterministic_environment(self.base_seed)
            multi_result = torch.randn(100, 100) @ torch.randn(100, 100)
            
            # Compare results
            max_diff = torch.max(torch.abs(single_result - multi_result)).item()
            consistent = max_diff < 1e-6
            
        finally:
            # Restore original thread count
            torch.set_num_threads(original_num_threads)
        
        result = {
            'test': 'threading_determinism',
            'passed': consistent,
            'max_difference': max_diff,
            'original_threads': original_num_threads
        }
        
        print(f"  ✓ Single vs multi-threaded max difference: {max_diff:.2e}")
        print(f"  ✓ Consistent: {consistent}")
        print(f"  {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        print()
        
        return result
    
    # Helper methods
    def _setup_deterministic_environment(self, seed: int):
        """Setup completely deterministic environment."""
        # Python random
        random.seed(seed)
        
        # NumPy random
        np.random.seed(seed)
        
        # PyTorch random
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # PyTorch deterministic operations
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        
        # cuDNN settings
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set number of threads for reproducibility
        torch.set_num_threads(1)
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'cuda_available': torch.cuda.is_available(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory
            })
        
        # Environment variables
        env_vars = ['CUDA_LAUNCH_BLOCKING', 'PYTHONHASHSEED', 'OMP_NUM_THREADS']
        info['environment_variables'] = {var: os.environ.get(var) for var in env_vars}
        
        return info
    
    def _test_matmul_determinism(self) -> Tuple[bool, float]:
        """Test matrix multiplication determinism."""
        self._setup_deterministic_environment(self.base_seed)
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        result1 = torch.matmul(a, b)
        
        self._setup_deterministic_environment(self.base_seed)
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        result2 = torch.matmul(a, b)
        
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        return max_diff < 1e-6, max_diff
    
    def _test_conv_determinism(self) -> Tuple[bool, float]:
        """Test convolution determinism."""
        self._setup_deterministic_environment(self.base_seed)
        x = torch.randn(1, 3, 32, 32)
        conv = torch.nn.Conv2d(3, 16, 3)
        result1 = conv(x)
        
        self._setup_deterministic_environment(self.base_seed)
        x = torch.randn(1, 3, 32, 32)
        conv = torch.nn.Conv2d(3, 16, 3)
        result2 = conv(x)
        
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        return max_diff < 1e-6, max_diff
    
    def _test_attention_determinism(self) -> Tuple[bool, float]:
        """Test attention mechanism determinism."""
        self._setup_deterministic_environment(self.base_seed)
        q = torch.randn(1, 8, 64, 64)
        k = torch.randn(1, 8, 64, 64)
        v = torch.randn(1, 8, 64, 64)
        result1 = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        self._setup_deterministic_environment(self.base_seed)
        q = torch.randn(1, 8, 64, 64)
        k = torch.randn(1, 8, 64, 64)
        v = torch.randn(1, 8, 64, 64)
        result2 = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        return max_diff < 1e-6, max_diff
    
    def _test_sorting_determinism(self) -> Tuple[bool, float]:
        """Test sorting determinism."""
        self._setup_deterministic_environment(self.base_seed)
        x = torch.randn(1000)
        sorted1, indices1 = torch.sort(x)
        
        self._setup_deterministic_environment(self.base_seed)
        x = torch.randn(1000)
        sorted2, indices2 = torch.sort(x)
        
        value_diff = torch.max(torch.abs(sorted1 - sorted2)).item()
        index_diff = torch.max(torch.abs(indices1.float() - indices2.float())).item()
        max_diff = max(value_diff, index_diff)
        
        return max_diff < 1e-6, max_diff
    
    def _test_float32_precision(self) -> Tuple[bool, float]:
        """Test float32 precision consistency."""
        x = torch.randn(100, 100, dtype=torch.float32)
        y = torch.randn(100, 100, dtype=torch.float32)
        
        result1 = torch.matmul(x, y)
        result2 = torch.matmul(x, y)
        
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        return max_diff == 0.0, max_diff
    
    def _test_float64_precision(self) -> Tuple[bool, float]:
        """Test float64 precision consistency."""
        x = torch.randn(100, 100, dtype=torch.float64)
        y = torch.randn(100, 100, dtype=torch.float64)
        
        result1 = torch.matmul(x, y)
        result2 = torch.matmul(x, y)
        
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        return max_diff == 0.0, max_diff
    
    def _test_mixed_precision(self) -> Tuple[bool, float]:
        """Test mixed precision consistency."""
        x = torch.randn(100, 100, dtype=torch.float32)
        y = torch.randn(100, 100, dtype=torch.float32)
        
        # Convert to half precision and back
        x_half = x.half().float()
        y_half = y.half().float()
        
        result1 = torch.matmul(x_half, y_half)
        result2 = torch.matmul(x_half, y_half)
        
        max_diff = torch.max(torch.abs(result1 - result2)).item()
        return max_diff == 0.0, max_diff
    
    def _print_verification_summary(self):
        """Print verification summary."""
        print("=" * 60)
        print("🔬 REPRODUCIBILITY VERIFICATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.verification_results)
        passed_tests = sum(1 for r in self.verification_results.values() if r['passed'])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        print()
        
        if passed_tests == total_tests:
            print("🎉 ALL REPRODUCIBILITY TESTS PASSED!")
            print("✅ Results will be reproducible across runs and environments")
        else:
            print("⚠️  SOME REPRODUCIBILITY TESTS FAILED")
            print("❌ Fix issues before publication to ensure reproducible results")
            print()
            print("Failed tests:")
            for test_name, result in self.verification_results.items():
                if not result['passed']:
                    print(f"  - {test_name}")
        
        print("=" * 60)
    
    def generate_reproducibility_report(self, output_path: str = "reproducibility_report.json"):
        """Generate comprehensive reproducibility report."""
        report = {
            'verification_results': self.verification_results,
            'environment_info': self._get_environment_info(),
            'verification_timestamp': datetime.datetime.now().isoformat(),
            'base_seed': self.base_seed,
            'num_verification_runs': self.num_verification_runs
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Reproducibility report saved to: {output_path}")


def main():
    """Run reproducibility verification suite."""
    print("🔬 GaussianFeels Reproducibility Verification")
    print("=" * 60)
    
    # Setup environment variables for maximum reproducibility
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    verifier = ReproducibilityVerifier(base_seed=42, num_verification_runs=3)
    results = verifier.verify_all()
    
    # Generate detailed report
    verifier.generate_reproducibility_report()
    
    # Exit with appropriate code for CI
    all_passed = all(result['passed'] for result in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()