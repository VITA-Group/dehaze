# Improved Techniques for Learning to Dehaze and Beyond: A Collective Study

## Introduction
This is the official codebase for our paper "Improved Techniques for Learning to Dehaze and Beyond: A Collective Study".

The paper reviews the collective endeavors by the team of authors in exploring two interlinked important tasks, based on the recently released REalistic Single Image DEhazing ([RESIDE](https://sites.google.com/view/reside-dehaze-datasets)) benchmark: i) single image dehazing as a low-level image restoration problem; ii) high-level visual understanding (e.g., object detection) from hazy images. For the first task, the authors investigated on a variety of loss functions, and found perception-driven loss to improve dehazing performance very notably. For the second task, the authors came up with multiple solutions including using more advanced modules in the dehazing-detection cascade, as well as domain-adaptive object detectors. In both tasks, our proposed solutions are verified to significantly advance the state-of-the-art performance.

## Code organization
Each individual software package and corresponding documentation are located under `code/PACKAGE_NAME`

## PAD-Net
See `code/pad_net`

## Domain adaptation for MaskRNN
See `code/adapt_maskrnn`

## Acknowledgements
This collective study was initially performed as a team project effort in the Machine Learning course ([CSCE 633, Spring 2018](http://people.tamu.edu/~atlaswang/18CSCE633.html)) of CSE@TAMU, taught by Dr. Zhangyang Wang. We acknowledge the Texas A\&M High Performance Research Computing (HPRC) for providing a part of the computing resources used in this research.

## Contact
- Yu Liu: yliu129@tamu.edu
- Guanlong Zhao: gzhao@tamu.edu
- Boyuan Gong
- Yang Li
- Ritu Raj
- Niraj Goel
- Satya Kesav
- Sandeep Gottimukkala
- Zhangyang Wang: atlaswang@tamu.edu
- Wenqi Ren
- Dacheng Tao

## Citation
    @article{liu2018dehaze,
      title={Improved Techniques for Learning to Dehaze and Beyond: A Collective Studys},
      author={Yu Liu, Guanlong Zhao, Boyuan Gong, Yang Li, Ritu Raj, Niraj Goel, Satya Kesav, Sandeep Gottimukkala, Zhangyang Wang, Wenqi Ren, Dacheng Tao},
      journal={TBD},
      year={2018}
    }
