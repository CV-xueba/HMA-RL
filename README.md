# Learnable Adaptive Bilateral Filter for Improved Generalization in Single Image Super-Resolution

This repository contains the official implementation of the following paper:

## Abstract
In the evolving landscape of computer vision, Single Image Super-Resolution (SISR) has emerged as a crucial area of research, largely propelled by the advancements in deep learning. 
This paper addresses the formidable challenges faced by state-of-the-art SISR models trained on synthetic datasets utilizing a simplistic degraded kernel, such as the bicubic kernel, when applied to real-world scenarios. The observed rapid decline in performance of the SR model is attributed to distribution differences in low-resolution images between the training and application stages. To alleviate the distribution gap, we introduce an adaptive bilateral filter-based preprocessing operator. This operator is trained using a Hierarchical Multi-Agent Reinforcement Learning framework in a weakly supervised manner. To mitigate potential challenges associated with the lazy agent issues within the framework, we propose the incorporation of pixel-level and global rewards. Building upon the proposed preprocessing operator, we formulate a two-stage SR network, yielding substantial improvements in generalization and achieving state-of-the-art perceptual quality performance on real-world datasets.


## Getting Started

To evaluate the performance and effectiveness of our proposed method, follow the steps below:

### Testing Individual Stages

To test each stage of our framework individually:

```bash
# Test the HMARL stage
cd HMARL
python test.py

# Test the SRModel stage
cd SRModel
python test.py
```

### Conducting End-to-End Tests

For a comprehensive end-to-end evaluation:

```bash
python inference.py
```

## Authors

- **Wenhao Guo** - *Initial work* - [Wenhao Guo](https://github.com/whguo97)

## Acknowledgments

A special thanks to all contributors and researchers in the SISR domain whose insights and work laid the foundation for this project.

[comment]: <> (## License)

[comment]: <> (This project is shared under the MIT License, providing wide-ranging flexibility for usage, modification, and distribution. For more details, refer to [LICENSE.md]&#40;link&#41;.)


