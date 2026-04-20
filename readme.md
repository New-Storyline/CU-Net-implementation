# Implementation of the CU-Net approach
## Overview
This repository contains a custom implementation of the approach described in the original CU-Net paper:
**"CU-Net: LiDAR Depth-Only Completion With Coupled U-Net"** *Yufei Wang, Yuchao Dai, Qi Liu, Peng Yang, Jiadai Sun, and Bo Li.* IEEE Robotics and Automation Letters (2022).  
[Paper DOI](https://doi.org/10.48550/arXiv.2210.14898) | [Original Repository](https://github.com/YufeiWang777/CU-Net)

CU-Net is a novel architecture designed for LiDAR depth completion tasks, which aims to predict dense depth maps from sparse LiDAR data. The model consists of two coupled U-Net structures that work together to enhance the depth completion performance. This implementation will be built using PyTorch Lightning for efficient training and modularity.

You can find a detailed description of the CU-Net architecture in the [CU-Net Architecture Documentation](model_arcs/CU-Net_arcitecture.md).

## To Do
Currently, this repository is a work in progress. The following tasks are planned for implementation:
- [ ] Implement the CU-Net model architecture based on [Original Repository](https://github.com/YufeiWang777/CU-Net).
- [ ] Implement the training loop and loss functions as described in the paper via PyTorch Lightning.
- [ ] Model training and evaluation on the NYU Depth V2 dataset.
- [ ] Add documentation and usage examples.

## Acknowledgments / Citation

If you use this code in your academic work, please cite the original authors:

```bibtex
@ARTICLE{wang2022ral,
  author={Wang, Yufei and Dai, Yuchao and Liu, Qi and Yang, Peng and Sun, Jiadai and Li, Bo},
  journal={IEEE Robotics and Automation Letters}, 
  title={CU-Net: LiDAR Depth-Only Completion With Coupled U-Net}, 
  year={2022},
  volume={7},
  number={4},
  pages={11476-11483},
  doi={10.1109/LRA.2022.3201193}
}