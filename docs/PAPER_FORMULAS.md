# LaTeX Formulas for Academic Paper

This document contains copy-paste ready LaTeX formulas for writing academic papers about GaussianFeels.

## Core Algorithm Formulas

### 3D Gaussian Definition
```latex
G(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{(2\pi)^{3/2}|\boldsymbol{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)\right)
```

### Covariance Factorization
```latex
\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T
```

### Volume Rendering Equation
```latex
C(\mathbf{p}) = \sum_{i=1}^N c_i(\mathbf{d}) \alpha_i(\mathbf{p}) T_i(\mathbf{p})
```

### Transmittance Computation
```latex
T_i(\mathbf{p}) = \prod_{j=1}^{i-1}(1-\alpha_j(\mathbf{p}))
```

### Multi-Modal Loss Function
```latex
\mathcal{L} = \lambda_{rgb} \mathcal{L}_{rgb} + \lambda_{depth} \mathcal{L}_{depth} + \lambda_{tactile} \mathcal{L}_{tactile} + \lambda_{reg} \mathcal{L}_{reg}
```

## Method Section Formulas

### Problem Formulation
```latex
\min_{\{\boldsymbol{\mu}_i, \mathbf{q}_i, \mathbf{s}_i, \alpha_i, \mathbf{c}_i\}} \sum_{j=1}^{N_v} \mathcal{L}(\mathbf{I}_j, \pi(\{\Theta_i\}, \mathbf{T}_j))
```

### Gaussian Parameters
```latex
\Theta_i = \{\boldsymbol{\mu}_i \in \mathbb{R}^3, \mathbf{q}_i \in \mathbb{S}^3, \mathbf{s}_i \in \mathbb{R}^3, \alpha_i \in [0,1], \mathbf{c}_i \in \mathbb{R}^{48}\}
```

### 2D Projection
```latex
\boldsymbol{\Sigma}_{2D} = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma}_{3D} \mathbf{W}^T \mathbf{J}^T
```

### Gaussian Weight at Pixel
```latex
\alpha_i(\mathbf{p}) = \alpha_i \exp\left(-\frac{1}{2}(\mathbf{p}-\boldsymbol{\mu}_{2D,i})^T \boldsymbol{\Sigma}_{2D,i}^{-1} (\mathbf{p}-\boldsymbol{\mu}_{2D,i})\right)
```

## Multi-Modal Components

### RGB Loss
```latex
\mathcal{L}_{rgb} = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{p} \in \mathcal{P}} \|C_{rendered}(\mathbf{p}) - C_{target}(\mathbf{p})\|_1
```

### Depth Loss  
```latex
\mathcal{L}_{depth} = \frac{1}{|\mathcal{P}_{valid}|} \sum_{\mathbf{p} \in \mathcal{P}_{valid}} \|D_{rendered}(\mathbf{p}) - D_{target}(\mathbf{p})\|_1
```

### Tactile Loss
```latex
\mathcal{L}_{tactile} = \sum_{k=1}^{N_c} w_k \|\mathbf{d}_k(\mathbf{p}_c^k) - \mathbf{d}_{target}^k\|_2^2
```

### Contact Point Transformation
```latex
\mathbf{p}_w = \mathbf{T}_{sensor}^{world} \mathbf{p}_{sensor}
```

## Optimization Details

### Densification Condition
```latex
\|\nabla_{\boldsymbol{\mu}_i} \mathcal{L}\|_2 > \tau_{grad} \text{ and } \max(\mathbf{s}_i) > \tau_{scale}
```

### Gaussian Splitting
```latex
\begin{align}
\boldsymbol{\mu}_{new} &= \boldsymbol{\mu}_i \pm \frac{\mathbf{s}_i}{2} \cdot \mathbf{v}_{max} \\
\mathbf{s}_{new} &= \mathbf{s}_i / 1.6
\end{align}
```

### Pruning Condition
```latex
\alpha_i < \tau_{prune} = 0.005
```

## Spherical Harmonics

### Color Function
```latex
c(\mathbf{d}) = \sum_{l=0}^{3} \sum_{m=-l}^{l} c_l^m Y_l^m(\theta, \phi)
```

### Basis Functions (Key Examples)
```latex
\begin{align}
Y_0^0 &= \frac{1}{2\sqrt{\pi}} \\
Y_1^{-1} &= \sqrt{\frac{3}{4\pi}} y, \quad Y_1^0 = \sqrt{\frac{3}{4\pi}} z, \quad Y_1^1 = \sqrt{\frac{3}{4\pi}} x \\
Y_2^0 &= \sqrt{\frac{5}{16\pi}} (3z^2-1)
\end{align}
```

## SE(3) Pose Optimization

### Pose Update
```latex
\mathbf{T} = \exp(\boldsymbol{\xi}^\wedge) \mathbf{T}_0
```

### SE(3) Exponential Map
```latex
\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix}
\exp(\boldsymbol{\phi}^\wedge) & \mathbf{V}\boldsymbol{\rho} \\
\mathbf{0}^T & 1
\end{bmatrix}
```

### Bundle Adjustment
```latex
\min_{\{\mathbf{T}_j\}, \{\Theta_i\}} \sum_{j=1}^{N_v} \sum_{\mathbf{p} \in \mathcal{P}_j} \rho\left(\|I_j(\mathbf{p}) - \pi(\mathbf{T}_j, \{\Theta_i\}, \mathbf{p})\|_2^2\right)
```

## Evaluation Metrics

### PSNR
```latex
\text{PSNR} = 10 \log_{10} \frac{\text{MAX}^2}{\text{MSE}}
```

### SSIM
```latex
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
```

### LPIPS
```latex
\text{LPIPS} = \frac{1}{HW} \sum_{h,w} \|\phi_h(\mathbf{x}_{hw}) - \phi_h(\mathbf{y}_{hw})\|_2^2
```

## Algorithm Pseudocode (for paper)

### Training Loop
```
Algorithm 1: GaussianFeels Training
Input: Multi-modal dataset D = {(I_rgb, I_depth, T_tactile, P_pose)}
Output: Optimized Gaussian field {Θ_i}

1: Initialize Gaussian field from point cloud
2: for epoch = 1 to N_epochs do
3:    for batch in D do
4:        // Forward pass
5:        I_rendered = Render({Θ_i}, P_pose)
6:        
7:        // Multi-modal loss
8:        L = λ_rgb * L_rgb + λ_depth * L_depth + λ_tactile * L_tactile
9:        
10:       // Backward pass
11:       ∇Θ = BackwardPass(L)
12:       
13:       // Adaptive maintenance
14:       if iter % densify_interval == 0 then
15:           DensifyAndPrune({Θ_i}, ∇Θ)
16:       end if
17:       
18:       // Parameter update
19:       {Θ_i} = Adam({Θ_i}, ∇Θ, lr)
20:   end for
21: end for
```

### Densification Algorithm
```
Algorithm 2: Adaptive Densification
Input: Gaussians {Θ_i}, gradients {∇μ_i}
Output: Updated Gaussian field

1: for i = 1 to N_gaussians do
2:    if ||∇μ_i||_2 > τ_grad then
3:        if max(s_i) > τ_scale then
4:            Split(Θ_i)  // Large Gaussian -> split
5:        else
6:            Clone(Θ_i)  // Small Gaussian -> clone
7:        end if
8:    end if
9:    
10:   if α_i < τ_prune then
11:       Remove(Θ_i)  // Low opacity -> prune
12:   end if
13: end for
```

## Table Templates

### Quantitative Results Template
```latex
\begin{table}[t]
\centering
\caption{Quantitative comparison on FeelSight dataset.}
\begin{tabular}{lccccc}
\toprule
Method & PSNR ↑ & SSIM ↑ & LPIPS ↓ & FPS ↑ & Memory \\
\midrule
NeRF & 31.01 & 0.947 & 0.163 & 0.04 & 8GB \\
3D-GS & 33.18 & 0.962 & 0.104 & 134.0 & 2GB \\
InstantNGP & 32.45 & 0.954 & 0.127 & 25.0 & 4GB \\
\textbf{Ours (Vision)} & \textbf{X.XX} & \textbf{X.XXX} & \textbf{X.XXX} & \textbf{XX.X} & \textbf{XGB} \\
\textbf{Ours (Multi-modal)} & \textbf{X.XX} & \textbf{X.XXX} & \textbf{X.XXX} & \textbf{XX.X} & \textbf{XGB} \\
\bottomrule
\end{tabular}
\label{tab:quantitative}
\end{table}
```

### Ablation Study Template
```latex
\begin{table}[t]
\centering
\caption{Ablation study on multi-modal components.}
\begin{tabular}{lcccc}
\toprule
Configuration & PSNR & SSIM & LPIPS & Contact Acc. \\
\midrule
Vision only & X.XX & X.XXX & X.XXX & - \\
+ Depth & X.XX & X.XXX & X.XXX & - \\
+ Tactile & X.XX & X.XXX & X.XXX & XX.X\% \\
+ Contact constraints & X.XX & X.XXX & X.XXX & XX.X\% \\
\textbf{Full method} & \textbf{X.XX} & \textbf{X.XXX} & \textbf{X.XXX} & \textbf{XX.X\%} \\
\bottomrule
\end{tabular}
\label{tab:ablation}
\end{table}
```

## Figure Captions

### Architecture Figure
```latex
\caption{Overview of GaussianFeels architecture. Our method combines RGB-D cameras and tactile sensors to reconstruct contact-aware 3D Gaussian fields. The multi-modal fusion enables accurate surface reconstruction in manipulation-relevant regions.}
```

### Results Figure
```latex
\caption{Qualitative comparison on the FeelSight dataset. Our multi-modal approach (d) produces more accurate reconstructions than vision-only methods (a-c), particularly in contact regions highlighted by tactile sensing.}
```

### Ablation Figure
```latex
\caption{Ablation study visualization. Adding tactile information (c) significantly improves reconstruction quality in contact regions compared to vision-only (a) and depth-only (b) approaches.}
```

## Mathematical Symbols Reference

### Notation Table
```latex
\begin{table}[h]
\centering
\caption{Mathematical notation used throughout the paper.}
\begin{tabular}{cl}
\toprule
Symbol & Description \\
\midrule
$\mathbf{x} \in \mathbb{R}^3$ & 3D world coordinates \\
$\boldsymbol{\mu}_i \in \mathbb{R}^3$ & Gaussian center position \\
$\boldsymbol{\Sigma}_i \in \mathbb{R}^{3 \times 3}$ & Gaussian covariance matrix \\
$\mathbf{R}_i \in SO(3)$ & Rotation matrix \\
$\mathbf{q}_i \in \mathbb{S}^3$ & Unit quaternion \\
$\mathbf{s}_i \in \mathbb{R}^3$ & Log-space scale parameters \\
$\alpha_i \in [0,1]$ & Gaussian opacity \\
$\mathbf{c}_i \in \mathbb{R}^{48}$ & Spherical harmonics coefficients \\
$\mathbf{T} \in SE(3)$ & Rigid body transformation \\
$\boldsymbol{\xi} \in \mathfrak{se}(3)$ & SE(3) tangent space vector \\
$C(\mathbf{p})$ & Rendered color at pixel $\mathbf{p}$ \\
$D(\mathbf{p})$ & Rendered depth at pixel $\mathbf{p}$ \\
$T_i(\mathbf{p})$ & Transmittance for Gaussian $i$ \\
$Y_l^m(\theta, \phi)$ & Spherical harmonic basis function \\
$\mathbf{p}_c$ & Tactile contact point \\
$\mathbf{d}_{tactile}$ & Tactile depth measurement \\
\bottomrule
\end{tabular}
\end{table}
```

## Common LaTeX Packages Needed

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}  % for bold math
\usepackage{booktabs}  % for nice tables
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage{algorithmic}
```

## Bibliography Entries

### Key References
```bibtex
@article{kerbl2023gaussian,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}

@article{zwicker2002ewa,
  title={EWA splatting},
  author={Zwicker, Matthias and Pfister, Hanspeter and Van Baar, Jeroen and Gross, Markus},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={8},
  number={3},
  pages={223--238},
  year={2002}
}

@article{ramamoorthi2001efficient,
  title={An efficient representation for irradiance environment maps},
  author={Ramamoorthi, Ravi and Hanrahan, Pat},
  journal={SIGGRAPH},
  pages={497--506},
  year={2001}
}

@article{mildenhall2020nerf,
  title={NeRF: Representing scenes as neural radiance fields for view synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  journal={European Conference on Computer Vision (ECCV)},
  pages={405--421},
  year={2020}
}

@article{yuan2017gelsight,
  title={GelSight: High-resolution robot tactile sensors for estimating geometry and force},
  author={Yuan, Wenzhen and Dong, Siyuan and Adelson, Edward H},
  journal={Sensors},
  volume={17},
  number={12},
  pages={2762},
  year={2017}
}

@article{calandra2018feeling,
  title={More than a feeling: Learning to grasp and regrasp using vision and touch},
  author={Calandra, Roberto and Owens, Andrew and Jayaraman, Dinesh and Lin, Justin and Yuan, Wenzhen and Malik, Jitendra and Adelson, Edward H and Levine, Sergey},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={4},
  pages={3300--3307},
  year={2018}
}

@article{shoemake1985animating,
  title={Animating rotation with quaternion curves},
  author={Shoemake, Ken},
  journal={SIGGRAPH Computer Graphics},
  volume={19},
  number={3},
  pages={245--254},
  year={1985}
}
```