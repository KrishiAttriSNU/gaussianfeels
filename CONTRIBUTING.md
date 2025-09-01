# Contributing to GaussianFeels

We welcome contributions to GaussianFeels! This document outlines the process for contributing to this academic research project and maintaining the high standards expected for publication-quality code.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Types](#contribution-types)
- [Coding Standards](#coding-standards)
- [Academic Standards](#academic-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Mathematical Correctness](#mathematical-correctness)
- [Reproducibility Requirements](#reproducibility-requirements)

## Code of Conduct

This project adheres to academic integrity and research ethics principles:

- **Academic Honesty**: All contributions must be original work or properly attributed
- **Collaborative Spirit**: Constructive feedback and respectful discussion
- **Reproducibility**: All changes must maintain reproducible research standards
- **Quality Assurance**: Code must meet academic publication standards

## Getting Started

### Prerequisites

1. **Academic Background**: Understanding of 3D computer vision, Gaussian splatting, and multi-modal sensing
2. **Technical Skills**: Python 3.8+, PyTorch, CUDA programming, mathematical optimization
3. **Research Ethics**: Familiarity with academic research practices and peer review standards

### Development Environment

```bash
# Clone the repository
git clone https://github.com/username/gaussianfeels.git
cd gaussianfeels

# Create development environment
conda create -n gaussianfeels-dev python=3.11
conda activate gaussianfeels-dev

# Install in development mode
pip install -e ".[dev,docs,gpu]"

# Install pre-commit hooks
pre-commit install
```

## Contribution Types

### 1. **Core Algorithm Improvements**
- Mathematical optimizations
- Numerical stability enhancements  
- Performance optimizations
- **Requirements**: Mathematical proofs, comprehensive testing, performance validation

### 2. **Multi-Modal Integration**
- New sensor modalities
- Fusion algorithm enhancements
- Calibration improvements
- **Requirements**: Sensor specifications, calibration procedures, validation datasets

### 3. **Academic Validation**
- Reproducibility improvements
- Benchmarking scripts
- Evaluation metrics
- **Requirements**: Statistical significance testing, multiple runs, confidence intervals

### 4. **Documentation & Research**
- Mathematical documentation
- Algorithm explanations  
- Research methodology
- **Requirements**: Academic writing standards, proper citations, LaTeX formatting

## Coding Standards

### Code Quality
```python
# Use type hints consistently
def render_gaussians(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor
) -> torch.Tensor:
    """
    Render 3D Gaussians using volumetric splatting.
    
    Mathematical Foundation:
    Each Gaussian is defined as:
    G(x; μᵢ, Σᵢ) = (2π)^(-3/2) |Σᵢ|^(-1/2) exp(-1/2 (x-μᵢ)ᵀ Σᵢ⁻¹ (x-μᵢ))
    
    Args:
        positions: Gaussian centers μᵢ ∈ ℝ³ [N, 3]
        rotations: Unit quaternions qᵢ ∈ S³ [N, 4] 
        scales: Log-space scales sᵢ ∈ ℝ³ [N, 3]
        opacities: Sigmoid-space opacities αᵢ ∈ [0,1] [N, 1]
        
    Returns:
        Rendered image tensor [H, W, 3]
        
    References:
        Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" SIGGRAPH 2023
    """
```

### Mathematical Documentation
- **All algorithms** must include mathematical foundations
- **Cite relevant papers** in docstrings
- **Explain parameter choices** with mathematical justification
- **Include numerical stability considerations**

### Code Organization
```
gaussianfeels/
├── core/              # Core algorithms with mathematical docs
├── render/            # Rendering pipeline
├── optimization/      # Training and optimization
├── evaluation/        # Academic evaluation tools
├── validation/        # Mathematical validation scripts
└── papers/            # LaTeX formulas and references
```

## Academic Standards

### 1. **Mathematical Correctness**
Every mathematical operation must be:
- **Theoretically sound**: Based on established mathematical principles
- **Numerically stable**: Tested with extreme values and edge cases
- **Properly documented**: With derivations and references
- **Validated**: Against known analytical solutions where possible

### 2. **Experimental Rigor**
- **Multiple runs**: All experiments must include variance statistics  
- **Statistical testing**: Significance tests for claimed improvements
- **Baseline comparisons**: Fair comparison with state-of-the-art methods
- **Ablation studies**: Systematic analysis of component contributions

### 3. **Reproducibility**
```python
# All experiments must be reproducible
def setup_reproducible_environment(seed: int = 42) -> None:
    """Setup deterministic environment for reproducible research."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Testing Requirements

### Unit Tests
```python
class TestGaussianMath:
    """Test mathematical correctness of Gaussian operations."""
    
    def test_covariance_decomposition(self):
        """Test Σ = R S Sᵀ Rᵀ decomposition."""
        # Generate random valid parameters
        rotations = random_unit_quaternions(100)
        scales = torch.exp(torch.randn(100, 3))  # Log-space parameterization
        
        # Compute covariance matrices
        covariances = compute_covariance_matrices(rotations, scales)
        
        # Verify positive definiteness
        eigenvals = torch.linalg.eigvals(covariances)
        assert torch.all(eigenvals.real > 1e-6), "Covariance matrices must be positive definite"
        
        # Verify reconstruction
        R = quaternion_to_rotation_matrix(rotations)
        S = torch.diag_embed(scales)
        reconstructed = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        torch.testing.assert_close(covariances, reconstructed, rtol=1e-5)
```

### Integration Tests
- **End-to-end pipeline tests**
- **Multi-modal data processing**
- **Performance regression tests**
- **Memory leak detection**

### Mathematical Validation
```python
def test_spherical_harmonics_orthogonality():
    """Validate spherical harmonics orthogonality properties."""
    # Test orthogonality: ∫ Yₗᵐ(θ,φ) Yₗ'ᵐ'(θ,φ) dΩ = δₗₗ' δₘₘ'
    pass

def test_volume_rendering_consistency():
    """Test volume rendering equation implementation."""
    # Verify: C(r) = ∫ T(t) σ(r(t)) c(r(t)) dt
    # Where T(t) = exp(-∫₀ᵗ σ(r(s)) ds)
    pass
```

## Documentation Standards

### Mathematical Notation
Use consistent mathematical notation throughout:

- **Vectors**: Bold lowercase (e.g., **r**, **x**)
- **Matrices**: Bold uppercase (e.g., **R**, **Σ**)  
- **Scalars**: Italic (e.g., *α*, *σ*)
- **Sets**: Blackboard bold (e.g., ℝ³, SO(3))
- **Functions**: Regular font (e.g., G(x), f(θ))

### LaTeX Documentation
```latex
\begin{align}
\text{Gaussian: } G(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) &= \frac{1}{(2\pi)^{3/2}|\boldsymbol{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)\right) \\
\text{Covariance: } \boldsymbol{\Sigma}_i &= \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T \\
\text{Rendering: } C(\mathbf{r}) &= \sum_{i=1}^N c_i(\mathbf{d}) \alpha_i(\mathbf{p}) T_i(\mathbf{p})
\end{align}
```

## Pull Request Process

### 1. **Pre-submission Checklist**
- [ ] Code follows academic standards and style guide
- [ ] All tests pass (unit, integration, mathematical validation)
- [ ] Documentation updated with mathematical foundations  
- [ ] Performance impact assessed and documented
- [ ] Reproducibility verified on clean environment
- [ ] Academic citations updated and properly formatted

### 2. **Pull Request Template**
```markdown
## Mathematical Foundation
Brief description of mathematical principles underlying the changes.

## Experimental Validation  
Description of tests performed to validate correctness and performance.

## Academic Impact
How these changes contribute to the research goals and publication quality.

## Reproducibility
Steps to reproduce results and verify correctness.
```

### 3. **Review Process**
- **Technical Review**: Code quality, mathematical correctness
- **Academic Review**: Research contribution, experimental rigor  
- **Reproducibility Review**: Independent verification of results
- **Documentation Review**: Academic writing standards, proper citations

## Mathematical Correctness

### Validation Requirements
Every mathematical implementation must include:

1. **Analytical Tests**: Comparison with known analytical solutions
2. **Numerical Tests**: Stability under extreme conditions
3. **Property Tests**: Verification of mathematical properties
4. **Convergence Tests**: Optimization algorithm convergence

### Example: Gaussian Rendering Validation
```python
def validate_gaussian_rendering():
    """Validate Gaussian rendering against analytical solutions."""
    # Test 1: Single Gaussian at origin should integrate to 1
    # ∫∫∫ G(x; 0, I) dx = 1
    
    # Test 2: Orthogonal Gaussians should not interfere
    # ∫ G₁(x) G₂(x) dx ≈ 0 for well-separated Gaussians
    
    # Test 3: Volume rendering should be order-independent  
    # Different sorting orders should produce same result
```

## Reproducibility Requirements

### Environment Specification
```python
# Always include environment specification
def get_environment_info() -> dict:
    """Get complete environment specification for reproducibility."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "random_seed": 42,
        "deterministic_algorithms": True,
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.now().isoformat()
    }
```

### Data Provenance
- **Dataset versions** must be explicitly specified
- **Preprocessing steps** must be documented and reproducible
- **Random splits** must be deterministic and documented
- **Evaluation protocols** must be standardized

## Questions and Support

### Academic Questions
- **Mathematical foundations**: Create issue with "math" label
- **Algorithm implementation**: Create issue with "algorithm" label  
- **Experimental design**: Create issue with "experiment" label

### Technical Support
- **Installation issues**: Check installation docs first
- **Performance issues**: Include profiling information
- **Reproducibility issues**: Include complete environment specification

### Research Collaboration
- **New research directions**: Open discussion issue
- **Dataset contributions**: Follow data contribution guidelines
- **Evaluation protocols**: Propose standardized evaluation procedures

---

**Remember**: This is an academic research project. All contributions should meet the standards expected for top-tier conference and journal publications. Quality over quantity, mathematical rigor over quick fixes.