# Mathematical Framework for GaussianFeels

This document provides a comprehensive mathematical reference for the GaussianFeels system, suitable for academic paper writing and theoretical understanding.

## Table of Contents
1. [3D Gaussian Representation](#3d-gaussian-representation)
2. [Volume Rendering](#volume-rendering)
3. [Multi-Modal Loss Functions](#multi-modal-loss-functions)
4. [Pose Optimization](#pose-optimization)
5. [Adaptive Field Maintenance](#adaptive-field-maintenance)
6. [Spherical Harmonics](#spherical-harmonics)
7. [Numerical Stability](#numerical-stability)
8. [Tactile Integration](#tactile-integration)

---

## 3D Gaussian Representation

### Gaussian Definition
Each 3D Gaussian primitive is defined as:

```latex
G(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{(2\pi)^{3/2}|\boldsymbol{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)\right)
```

**Where:**
- $\mathbf{x} \in \mathbb{R}^3$: 3D world coordinates
- $\boldsymbol{\mu}_i \in \mathbb{R}^3$: Gaussian center (position)
- $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3 \times 3}$: Covariance matrix (shape/orientation)

### Covariance Parameterization
For numerical stability and optimization, we factorize the covariance matrix:

```latex
\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T
```

**Where:**
- $\mathbf{R}_i \in SO(3)$: Rotation matrix from unit quaternion $\mathbf{q}_i$
- $\mathbf{S}_i = \text{diag}(\exp(s_i^x), \exp(s_i^y), \exp(s_i^z))$: Scale matrix (log-parameterized)

### Quaternion to Rotation Matrix
The rotation matrix $\mathbf{R}_i$ is computed from unit quaternion $\mathbf{q}_i = [w, x, y, z]^T$:

```latex
\mathbf{R}_i = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
```

### Gaussian Parameters
Each Gaussian $i$ is parameterized by:
- **Position**: $\boldsymbol{\mu}_i \in \mathbb{R}^3$ (directly optimized)
- **Rotation**: $\mathbf{q}_i \in \mathbb{S}^3$ (unit quaternion)
- **Scale**: $\mathbf{s}_i \in \mathbb{R}^3$ (log-space for positive definiteness)
- **Opacity**: $\alpha_i \in [0,1]$ (sigmoid-activated from logit)
- **Color**: Spherical harmonics coefficients $\mathbf{c}_i \in \mathbb{R}^{48}$ (degree 0-3)

---

## Volume Rendering

### 2D Projection
3D Gaussians are projected to 2D screen space using perspective projection:

```latex
\boldsymbol{\Sigma}_{2D} = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma}_{3D} \mathbf{W}^T \mathbf{J}^T
```

**Where:**
- $\mathbf{J} \in \mathbb{R}^{2 \times 3}$: Jacobian of perspective projection
- $\mathbf{W} \in \mathbb{R}^{3 \times 3}$: World-to-camera transformation
- $\boldsymbol{\Sigma}_{3D}$: 3D covariance matrix

### 2D Gaussian Weight
The 2D Gaussian weight at pixel $\mathbf{p} = [u, v]^T$ is:

```latex
\alpha_i(\mathbf{p}) = \alpha_i \exp\left(-\frac{1}{2}(\mathbf{p}-\boldsymbol{\mu}_{2D,i})^T \boldsymbol{\Sigma}_{2D,i}^{-1} (\mathbf{p}-\boldsymbol{\mu}_{2D,i})\right)
```

### Volume Rendering Equation
The final pixel color is computed using front-to-back alpha compositing:

```latex
C(\mathbf{p}) = \sum_{i=1}^N c_i(\mathbf{d}) \alpha_i(\mathbf{p}) T_i(\mathbf{p})
```

**Where:**
- $c_i(\mathbf{d})$: View-dependent color from spherical harmonics
- $\alpha_i(\mathbf{p})$: 2D Gaussian weight at pixel $\mathbf{p}$
- $T_i(\mathbf{p}) = \prod_{j=1}^{i-1}(1-\alpha_j(\mathbf{p}))$: Transmittance

### Depth Rendering
Depth at pixel $\mathbf{p}$ is computed analogously:

```latex
D(\mathbf{p}) = \sum_{i=1}^N d_i \alpha_i(\mathbf{p}) T_i(\mathbf{p})
```

**Where:**
- $d_i$: Depth of Gaussian $i$ along camera ray

---

## Multi-Modal Loss Functions

### Total Loss
The total training loss combines multiple modalities:

```latex
\mathcal{L} = \lambda_{rgb} \mathcal{L}_{rgb} + \lambda_{depth} \mathcal{L}_{depth} + \lambda_{tactile} \mathcal{L}_{tactile} + \lambda_{reg} \mathcal{L}_{reg}
```

### RGB Loss
Photometric consistency loss using L1 norm:

```latex
\mathcal{L}_{rgb} = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{p} \in \mathcal{P}} \|C_{rendered}(\mathbf{p}) - C_{target}(\mathbf{p})\|_1
```

### Depth Loss
Geometric consistency loss on valid depth pixels:

```latex
\mathcal{L}_{depth} = \frac{1}{|\mathcal{P}_{valid}|} \sum_{\mathbf{p} \in \mathcal{P}_{valid}} \|D_{rendered}(\mathbf{p}) - D_{target}(\mathbf{p})\|_1
```

### Tactile Loss
Contact-aware surface reconstruction loss:

```latex
\mathcal{L}_{tactile} = \sum_{k=1}^{N_c} w_k \|\mathbf{d}_k(\mathbf{p}_c^k) - \mathbf{d}_{target}^k\|_2^2
```

**Where:**
- $\mathbf{p}_c^k$: Contact point $k$ in 3D space
- $\mathbf{d}_k$: Predicted deformation/depth at contact
- $\mathbf{d}_{target}^k$: Target tactile measurement
- $w_k$: Contact confidence weight

### Regularization Loss
Smoothness and sparsity regularization:

```latex
\mathcal{L}_{reg} = \lambda_{smooth} \sum_{i=1}^N \|\nabla \boldsymbol{\mu}_i\|_2^2 + \lambda_{sparse} \sum_{i=1}^N (1-\alpha_i)^2
```

---

## Pose Optimization

### SE(3) Parameterization
Object poses are parameterized in SE(3) using the tangent space:

```latex
\mathbf{T} = \exp(\boldsymbol{\xi}^\wedge) \mathbf{T}_0
```

**Where:**
- $\boldsymbol{\xi} \in \mathbb{R}^6$: Tangent space vector $[\boldsymbol{\rho}, \boldsymbol{\phi}]^T$
- $\boldsymbol{\rho} \in \mathbb{R}^3$: Translation component  
- $\boldsymbol{\phi} \in \mathbb{R}^3$: Rotation component (axis-angle)
- $\mathbf{T}_0 \in SE(3)$: Initial pose estimate

### SE(3) Exponential Map
The matrix exponential of $\boldsymbol{\xi}^\wedge$ is:

```latex
\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix}
\exp(\boldsymbol{\phi}^\wedge) & \mathbf{V}\boldsymbol{\rho} \\
\mathbf{0}^T & 1
\end{bmatrix}
```

**Where:**
- $\exp(\boldsymbol{\phi}^\wedge) = \mathbf{I} + \frac{\sin(\|\boldsymbol{\phi}\|)}{\|\boldsymbol{\phi}\|}\boldsymbol{\phi}^\wedge + \frac{1-\cos(\|\boldsymbol{\phi}\|)}{\|\boldsymbol{\phi}\|^2}(\boldsymbol{\phi}^\wedge)^2$
- $\mathbf{V} = \mathbf{I} + \frac{1-\cos(\|\boldsymbol{\phi}\|)}{\|\boldsymbol{\phi}\|^2}\boldsymbol{\phi}^\wedge + \frac{\|\boldsymbol{\phi}\|-\sin(\|\boldsymbol{\phi}\|)}{\|\boldsymbol{\phi}\|^3}(\boldsymbol{\phi}^\wedge)^2$

### Bundle Adjustment
Multi-view pose optimization with Gaussian field constraints:

```latex
\min_{\{\mathbf{T}_j\}, \{\Theta_i\}} \sum_{j=1}^{N_v} \sum_{\mathbf{p} \in \mathcal{P}_j} \rho\left(\|I_j(\mathbf{p}) - \pi(\mathbf{T}_j, \{\Theta_i\}, \mathbf{p})\|_2^2\right)
```

**Where:**
- $\mathbf{T}_j$: Camera pose for view $j$
- $\Theta_i$: Gaussian parameters for primitive $i$
- $\pi(\cdot)$: Rendering function
- $\rho(\cdot)$: Robust loss function (Huber)

---

## Adaptive Field Maintenance

### Gradient-Based Densification
Gaussians are split or cloned based on gradient magnitude:

```latex
\text{Split if: } \|\nabla_{\boldsymbol{\mu}_i} \mathcal{L}\|_2 > \tau_{grad} \text{ and } \max(\mathbf{s}_i) > \tau_{scale}
```

```latex
\text{Clone if: } \|\nabla_{\boldsymbol{\mu}_i} \mathcal{L}\|_2 > \tau_{grad} \text{ and } \max(\mathbf{s}_i) \leq \tau_{scale}
```

### Gaussian Splitting
When splitting Gaussian $i$:

```latex
\boldsymbol{\mu}_{new} = \boldsymbol{\mu}_i \pm \frac{\mathbf{s}_i}{2} \cdot \mathbf{v}_{max}
```

```latex
\mathbf{s}_{new} = \mathbf{s}_i / 1.6
```

**Where:**
- $\mathbf{v}_{max}$: Principal axis of largest scale
- Factor $1.6$ maintains volume conservation

### Opacity-Based Pruning
Gaussians with low opacity are removed:

```latex
\text{Prune if: } \alpha_i < \tau_{prune} = 0.005
```

### Large Gaussian Removal
Oversized Gaussians in camera view are removed:

```latex
\text{Remove if: } \max(\text{projected\_scale}_i) > 0.1 \cdot \min(W, H)
```

---

## Spherical Harmonics

### Color Representation
View-dependent color is represented using spherical harmonics up to degree 3:

```latex
c(\mathbf{d}) = \sum_{l=0}^{3} \sum_{m=-l}^{l} c_l^m Y_l^m(\theta, \phi)
```

**Where:**
- $\mathbf{d} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$: Viewing direction
- $Y_l^m(\theta, \phi)$: Spherical harmonic basis functions
- $c_l^m$: Spherical harmonic coefficients

### Basis Functions (Degree 0-3)
```latex
Y_0^0 = \frac{1}{2\sqrt{\pi}}
```

```latex
Y_1^{-1} = \sqrt{\frac{3}{4\pi}} y, \quad Y_1^0 = \sqrt{\frac{3}{4\pi}} z, \quad Y_1^1 = \sqrt{\frac{3}{4\pi}} x
```

```latex
Y_2^{-2} = \sqrt{\frac{15}{4\pi}} xy, \quad Y_2^{-1} = \sqrt{\frac{15}{4\pi}} yz
```

```latex
Y_2^0 = \sqrt{\frac{5}{16\pi}} (3z^2-1), \quad Y_2^1 = \sqrt{\frac{15}{4\pi}} xz, \quad Y_2^2 = \sqrt{\frac{15}{16\pi}} (x^2-y^2)
```

### DC and Non-DC Components
The color is split into view-independent (DC) and view-dependent components:

```latex
c(\mathbf{d}) = c_{DC} + \sum_{l=1}^{3} \sum_{m=-l}^{l} c_l^m Y_l^m(\mathbf{d})
```

---

## Numerical Stability

### Covariance Regularization
To ensure positive definiteness:

```latex
\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T + \epsilon \mathbf{I}
```

**Where:** $\epsilon = 10^{-7}$ prevents numerical issues.

### Scale Clamping
Gaussian scales are clamped to prevent degeneracy:

```latex
s_i^{min} = -10, \quad s_i^{max} = 10
```

Corresponding to scale factors in $[\exp(-10), \exp(10)] \approx [4.5 \times 10^{-5}, 2.2 \times 10^{4}]$.

### Quaternion Normalization
Unit quaternions are enforced at each optimization step:

```latex
\mathbf{q}_i \leftarrow \frac{\mathbf{q}_i}{\|\mathbf{q}_i\|_2}
```

### Opacity Activation
Opacity is parameterized using sigmoid activation:

```latex
\alpha_i = \sigma(\text{logit}_i) = \frac{1}{1 + \exp(-\text{logit}_i)}
```

---

## Tactile Integration

### Contact Point Transformation
Tactile contact points are transformed from sensor frame to world frame:

```latex
\mathbf{p}_w = \mathbf{T}_{sensor}^{world} \mathbf{p}_{sensor}
```

### Tactile Depth Prediction
The tactile sensor measures surface deformation:

```latex
d_{tactile} = \|\mathbf{p}_{contact} - \mathbf{p}_{surface}\|_2
```

### Multi-Modal Fusion Weight
The contribution of tactile vs. visual information is weighted by contact confidence:

```latex
w_{tactile} = \gamma \cdot \text{confidence}_{contact} + (1-\gamma) \cdot w_{base}
```

### Contact Surface Constraints
Tactile measurements provide constraints on surface normals and curvature:

```latex
\mathbf{n}_{predicted} \cdot \mathbf{n}_{tactile} > \cos(\tau_{normal})
```

**Where:** $\tau_{normal} = 30°$ is the maximum normal deviation tolerance.

---

## Optimization Details

### Learning Rate Schedule
Different parameter types use different learning rates:

```latex
\begin{align}
\eta_{\boldsymbol{\mu}} &= 2 \times 10^{-4} \\
\eta_{\mathbf{q}} &= 1 \times 10^{-3} \\
\eta_{\mathbf{s}} &= 5 \times 10^{-3} \\
\eta_{\alpha} &= 5 \times 10^{-2} \\
\eta_{SH} &= 2.5 \times 10^{-3}
\end{align}
```

### Adam Optimizer Parameters
```latex
\beta_1 = 0.9, \quad \beta_2 = 0.999, \quad \epsilon = 10^{-15}
```

### Gradient Clipping
Gradients are clipped to prevent instability:

```latex
\|\nabla \Theta\|_2 \leq \tau_{clip} = 1.0
```

---

## Implementation Notes

### Memory Efficiency
- Gaussians are stored in structure-of-arrays format for coalesced memory access
- Dynamic pruning maintains field size below memory limits
- Mixed precision training reduces memory footprint by 50%

### CUDA Optimization
- Tile-based rendering for efficient GPU utilization
- Shared memory usage for frequently accessed data
- Warp-level primitives for parallel reductions

### Deterministic Training
For reproducible results:
- Fixed random seeds across all components
- Deterministic CUDA operations enabled
- Consistent floating-point operations

---

**References:**
1. Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" SIGGRAPH 2023
2. Zwicker et al. "EWA Splatting" IEEE TVCG 2002
3. Ramamoorthi & Hanrahan "An Efficient Representation for Irradiance Environment Maps" SIGGRAPH 2001
4. Sola et al. "A micro Lie theory for state estimation in robotics" arXiv 2018