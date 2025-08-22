API Reference
=============

This section contains the complete API documentation for GaussianFeels.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   api/config
   api/trainer
   api/datasets
   api/optimization
   api/evaluation

Configuration
~~~~~~~~~~~~~

.. automodule:: gaussianfeels.config
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gaussianfeels.config.GaussianFeelsConfig
   :members:
   :undoc-members:
   :show-inheritance:

Trainer
~~~~~~~

.. automodule:: gaussianfeels.trainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gaussianfeels.trainer.GaussianTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Management
~~~~~~~~~~~~~~~~~~

.. automodule:: gaussianfeels.datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gaussianfeels.datasets.DatasetRegistry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gaussianfeels.datasets.BaseDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gaussianfeels.datasets.FrameData
   :members:
   :undoc-members:
   :show-inheritance:

Optimization
~~~~~~~~~~~~

.. automodule:: gaussianfeels.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
~~~~~~~~~~

.. automodule:: gaussianfeels.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Camera Pipeline
---------------

.. toctree::
   :maxdepth: 2

   api/camera/core
   api/camera/render
   api/camera/io

Core Components
~~~~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.core.gaussian_field
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: camera.gaussianfeels.core.densify_prune
   :members:
   :undoc-members:
   :show-inheritance:

Rendering Pipeline
~~~~~~~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.render.rasterizer
   :members:
   :undoc-members:
   :show-inheritance:

Input/Output
~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.io.rgbd_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: camera.gaussianfeels.io.segmentation
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
~~~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.loss.volumetric_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: camera.gaussianfeels.loss.tactile_loss
   :members:
   :undoc-members:
   :show-inheritance:

Tactile Integration
-------------------

.. toctree::
   :maxdepth: 2

   api/tactile/modules
   api/tactile/datasets

Tactile Modules
~~~~~~~~~~~~~~~

.. automodule:: tactile.gaussianfeels.modules.sensor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tactile.gaussianfeels.modules.allegro
   :members:
   :undoc-members:
   :show-inheritance:

Tactile Datasets
~~~~~~~~~~~~~~~~

.. automodule:: tactile.gaussianfeels.datasets.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Fusion Components
-----------------

.. automodule:: fusion
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Camera Utilities
~~~~~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.utils.camera_geometry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: camera.gaussianfeels.utils.pose_transforms
   :members:
   :undoc-members:
   :show-inheritance:

Performance Utilities
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.utils.performance_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Memory Management
~~~~~~~~~~~~~~~~~

.. automodule:: camera.gaussianfeels.memory.memory_monitor
   :members:
   :undoc-members:
   :show-inheritance:

Validation Framework
--------------------

Mathematical Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: validation.mathematical_validation
   :members:
   :undoc-members:
   :show-inheritance:

Reproducibility Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: validation.reproducibility_verification
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Functions
----------------------

Gaussian Operations
~~~~~~~~~~~~~~~~~~

.. py:function:: compute_covariance_matrices(rotations, scales)

   Compute 3D covariance matrices from rotation quaternions and scale vectors.
   
   **Mathematical Foundation:**
   
   .. math::
   
      \boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T
   
   :param torch.Tensor rotations: Unit quaternions [N, 4]
   :param torch.Tensor scales: Scale vectors [N, 3] (log-space)
   :returns: Covariance matrices [N, 3, 3]
   :rtype: torch.Tensor

.. py:function:: quaternion_to_rotation_matrix(quaternions)

   Convert unit quaternions to rotation matrices.
   
   **Mathematical Foundation:**
   
   For quaternion :math:`q = [w, x, y, z]^T`:
   
   .. math::
   
      \mathbf{R} = \begin{bmatrix}
      1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
      2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
      2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
      \end{bmatrix}
   
   :param torch.Tensor quaternions: Unit quaternions [N, 4]
   :returns: Rotation matrices [N, 3, 3]
   :rtype: torch.Tensor

Volume Rendering
~~~~~~~~~~~~~~~

.. py:function:: render_gaussians(positions, rotations, scales, opacities, colors, camera_params)

   Render 3D Gaussians using volumetric splatting.
   
   **Mathematical Foundation:**
   
   .. math::
   
      C(\mathbf{p}) = \sum_{i=1}^N c_i(\mathbf{d}) \alpha_i(\mathbf{p}) T_i(\mathbf{p})
   
   where :math:`T_i(\mathbf{p}) = \prod_{j=1}^{i-1}(1-\alpha_j(\mathbf{p}))`.
   
   :param torch.Tensor positions: Gaussian centers [N, 3]
   :param torch.Tensor rotations: Unit quaternions [N, 4]
   :param torch.Tensor scales: Log-space scales [N, 3]
   :param torch.Tensor opacities: Sigmoid-space opacities [N, 1]
   :param torch.Tensor colors: Spherical harmonics coefficients [N, 48]
   :param CameraParams camera_params: Camera parameters
   :returns: Rendered RGB image [H, W, 3] and depth [H, W, 1]
   :rtype: Tuple[torch.Tensor, torch.Tensor]

Spherical Harmonics
~~~~~~~~~~~~~~~~~~~

.. py:function:: evaluate_spherical_harmonics(directions, coefficients, max_degree=3)

   Evaluate spherical harmonics for view-dependent color.
   
   **Mathematical Foundation:**
   
   .. math::
   
      c(\mathbf{d}) = \sum_{l=0}^{3} \sum_{m=-l}^{l} c_l^m Y_l^m(\theta, \phi)
   
   :param torch.Tensor directions: Viewing directions [N, 3]
   :param torch.Tensor coefficients: SH coefficients [N, C]
   :param int max_degree: Maximum SH degree (default: 3)
   :returns: View-dependent colors [N, 3]
   :rtype: torch.Tensor

SE(3) Operations
~~~~~~~~~~~~~~~

.. py:function:: se3_exp(tangent_vectors)

   SE(3) exponential map from tangent space to group.
   
   **Mathematical Foundation:**
   
   .. math::
   
      \exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix}
      \exp(\boldsymbol{\phi}^\wedge) & \mathbf{V}\boldsymbol{\rho} \\
      \mathbf{0}^T & 1
      \end{bmatrix}
   
   :param torch.Tensor tangent_vectors: Tangent space vectors [N, 6]
   :returns: SE(3) transformation matrices [N, 4, 4]
   :rtype: torch.Tensor

Error Handling
--------------

.. py:exception:: GaussianFeelsError

   Base exception class for GaussianFeels errors.

.. py:exception:: ConfigurationError

   Raised when configuration parameters are invalid.

.. py:exception:: DatasetError

   Raised when dataset loading or processing fails.

.. py:exception:: RenderingError

   Raised when rendering operations fail.

.. py:exception:: OptimizationError

   Raised when optimization fails to converge.

Type Definitions
----------------

.. py:class:: CameraParams

   Camera parameter dataclass.
   
   :param torch.Tensor intrinsics: Camera intrinsic matrix [3, 3]
   :param torch.Tensor extrinsics: Camera extrinsic matrix [4, 4]
   :param int width: Image width
   :param int height: Image height
   :param float near: Near clipping plane
   :param float far: Far clipping plane

.. py:class:: RenderConfig

   Rendering configuration dataclass.
   
   :param int tile_size: Tile size for rendering (default: 16)
   :param float alpha_threshold: Alpha threshold for pruning (default: 0.005)
   :param bool enable_sh: Enable spherical harmonics (default: True)
   :param int max_sh_degree: Maximum SH degree (default: 3)

.. py:class:: GaussianConfig

   Gaussian field configuration dataclass.
   
   :param int max_gaussians: Maximum number of Gaussians (default: 300000)
   :param float densify_threshold: Gradient threshold for densification (default: 0.0002)
   :param float prune_threshold: Opacity threshold for pruning (default: 0.005)
   :param int densify_interval: Steps between densification (default: 100)