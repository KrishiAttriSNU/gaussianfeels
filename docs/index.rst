GaussianFeels Documentation
===========================

**Multi-Modal 3D Gaussian Splatting for Real-Time Object Reconstruction**

GaussianFeels is an academic-quality implementation of multi-modal 3D Gaussian splatting that combines RGB-D cameras and tactile sensors for contact-aware object reconstruction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   mathematical_framework
   api_reference
   tutorials
   examples
   development
   validation

Quick Start
-----------

Installation::

   pip install -e .

Basic Usage::

   from gaussianfeels import GaussianTrainer, GaussianFeelsConfig
   
   config = GaussianFeelsConfig(
       dataset="feelsight",
       object="contactdb_rubber_duck",
       log="00"
   )
   
   trainer = GaussianTrainer(config)
   trainer.train()

Key Features
------------

* **Multi-Modal Fusion**: Combines RGB-D and tactile sensing
* **Real-Time Performance**: Efficient volumetric rendering
* **Academic Quality**: Publication-ready implementation
* **Mathematical Rigor**: Comprehensive validation framework

Academic Framework
------------------

This implementation follows academic standards with:

* Complete mathematical documentation
* Reproducibility verification
* Statistical validation
* Performance benchmarking

Mathematical Foundation
-----------------------

The core 3D Gaussian representation:

.. math::

   G(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{(2\pi)^{3/2}|\boldsymbol{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)\right)

Volume rendering equation:

.. math::

   C(\mathbf{p}) = \sum_{i=1}^N c_i(\mathbf{d}) \alpha_i(\mathbf{p}) T_i(\mathbf{p})

where :math:`T_i(\mathbf{p}) = \prod_{j=1}^{i-1}(1-\alpha_j(\mathbf{p}))` is the transmittance.

Citation
--------

If you use GaussianFeels in your research, please cite::

   @article{gaussianfeels2024,
     title={GaussianFeels: Multi-Modal 3D Gaussian Splatting for Real-Time Object Reconstruction},
     author={Krishi Attri},
     institution={Soft Robotics and Bionics Lab, Seoul National University},
     journal={[Journal/Conference]},
     year={2024}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`