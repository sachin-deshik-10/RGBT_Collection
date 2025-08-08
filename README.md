
# RGB-T Research Collection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)](https://github.com/VisionVerse/Awesome-RGBT/pulls)
[![Stars](https://img.shields.io/github/stars/VisionVerse/Awesome-RGBT)](https://github.com/VisionVerse/Awesome-RGBT/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<div align="center">
  <img src="/thermal.png" alt="RGB-T Vision" width="400"/>
</div>

## 📖 Overview

This repository provides a comprehensive collection of **research papers, source codes, and datasets** for RGB-Thermal (RGB-T) computer vision tasks based on deep learning methodologies. Our curated list covers the latest advances in multi-modal vision research, combining visible and thermal imaging modalities.

### 🎯 Research Areas Covered

- **RGB-T Image Fusion** - Combining visible and thermal imagery for enhanced visual representation
- **RGB-T Salient Object Detection (SOD)** - Identifying prominent objects using multi-modal cues  
- **RGB-T Vehicle Detection** - Automotive applications with thermal-visible fusion
- **RGB-T Crowd Counting** - Population estimation using dual-modality sensing
- **RGB-T Pedestrian Detection** - Human detection with enhanced thermal information
- **RGB-T Semantic Segmentation** - Pixel-level scene understanding across modalities
- **RGB-T Object Tracking** - Temporal object following with thermal guidance
- **RGB-T Person Re-Identification** - Cross-modal person matching
- **RGB-T Image Alignment** - Registration between thermal and visible domains

### 📊 Collection Statistics

- **120+ Research Papers** included and continuously updated
- **Latest Publications** from top-tier conferences and journals
- **Open Source Implementations** with available code repositories
- **Benchmark Datasets** for reproducible research

---

> 🌟 **Please star this repository if you find it helpful!**
>
> 🔄 **This collection is actively maintained** - we welcome contributions via pull requests.

## 📑 Table of Contents

1. [RGB-T Fusion](#1-rgb-t-fusion)
2. [RGB-T Salient Object Detection](#2-rgb-t-salient-object-detection)
3. [RGB-T Vehicle Detection](#3-rgb-t-vehicle-detection)
4. [RGB-T Crowd Counting](#4-rgb-t-crowd-counting)
5. [RGB-T Pedestrian Detection](#5-rgb-t-pedestrian-detection)
6. [RGB-T Semantic Segmentation](#6-rgb-t-semantic-segmentation)
7. [RGB-T Tracking](#7-rgb-t-tracking)
8. [RGB-T Person Re-Identification](#8-rgb-t-person-re-identification)
9. [RGB-T Image Alignment](#9-rgb-t-image-alignment)

## 📅 Recent Updates

- **2025/07/23**: RGB-T Fusion +1, RGB-T Semantic Segmentation +1
- **2025/07/21**: RGB-T Crowd Counting +1
- **2025/07/16**: RGB-T Salient Object Detection +1, RGB-T Semantic Segmentation +1
- **2025/07/06**: RGB-T Fusion +2, RGB-T Tracking +2
- **2025/06/26**: RGB-T Salient Object Detection +2, RGB-T Tracking +1

---

## 1. RGB-T Fusion

**Latest Update:** 2025-07-23

| No. | Year | Model | Venue | Title | Resources |
|:-:|:-:|:-:|:-:|:--|:-:|
| 2 | 2025 | WSFM | TGRS | Weakly Supervised Cross Mixer for Infrared and Visible Image Fusion | [Paper](https://ieeexplore.ieee.org/abstract/document/11052679) • [Code](https://github.com/wangwenbo26/WSCM) |
| 1 | 2025 | T²EA | TCSVT | T²EA: Target-Aware Taylor Expansion Approximation Network for Infrared and Visible Image Fusion | [Paper](https://ieeexplore.ieee.org/document/10819442) • [Code](https://github.com/MysterYxby/T2EA) |
| 2 | 2024 | HitFusion | TMM | HitFusion: Infrared and Visible Image Fusion for High-Level Vision Tasks Using Transformer | [Paper](https://ieeexplore.ieee.org/document/10539339) |
| 1 | 2024 | CLIP | arXiv | From Text to Pixels: A Context-Aware Semantic Synergy Solution for Infrared and Visible Image Fusion | [Paper](https://arxiv.org/pdf/2401.00421.pdf) • [Analysis](https://zhuanlan.zhihu.com/p/676176433) |
| 2 | 2023 | RFVIF | Inf. Fusion | Feature dynamic alignment and refinement for infrared–visible image fusion: Translation robust fusion | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253523000519) • [Code](https://github.com/lhf12278/RFVIF) |
| 1 | 2023 | MURF ⭐ | TPAMI | MURF：Mutually Reinforcing Multi-Modal Image Registration and Fusion | [Paper](https://ieeexplore.ieee.org/document/10145843) • [Code](https://github.com/hanna-xu/MURF) |
| 1 | 2022 | RFNet ⭐ | CVPR | RFNet：Unsupervised Network for Mutually Reinforcing Multi-modal Image Registration and Fusion | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_RFNet_Unsupervised_Network_for_Mutually_Reinforcing_Multi-Modal_Image_Registration_and_CVPR_2022_paper.pdf) • [Code](https://github.com/hanna-xu/MURF) |
| 1 | 2020 | U2Fusion ⭐ | TPAMI | U2Fusion: A Unified Unsupervised Image Fusion Network | [Paper](https://ieeexplore.ieee.org/document/9151265) • [Code](https://github.com/hanna-xu/U2Fusion) |
| 1 | 2019 | Survey | Inf. Fusion | Infrared and visible image fusion methods and applications: A survey | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253517307972) |
| 1 | 2016 | GTF | Inf. Fusion | Infrared and visible image fusion via gradient transfer and total variation minimization | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S156625351630001X) • [Code](https://github.com/jiayi-ma/GTF) |

> ⭐ *Notable works from Professor Ma's research group at WHU*

### 📚 Fusion Resources

**Datasets:**

- [TNO Image Fusion Dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
- [KAIST Multispectral Dataset](https://soonminhwang.github.io/rgbt-ped-detection/)

**Evaluation Tools:**

- [Image Fusion Metrics](https://github.com/Linfeng-Tang/Image-Fusion-Evaluation-Metrics)

---

## 2. RGB-T Salient Object Detection

**Latest Update:** 2025-07-16

### Recent Publications (2024-2025)

| Year | Model | Venue | Title | Resources |
|:-:|:-:|:-:|:--|:-:|
| 2025 | AlignSal | TGRS | Efficient Fourier Filtering Network With Contrastive Learning for AAV-Based Unaligned Bimodal Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10975009) • [Code](https://github.com/JoshuaLPF/AlignSal) |
| 2025 | TwinsTNet | TIP | TwinsTNet: Broad-View Twins Transformer Network for Bi-Modal Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/10982382) • [Code](https://github.com/JoshuaLPF/TwinsTNet) |
| 2025 | KAN-SAM | arXiv | KAN-SAM: Kolmogorov-Arnold Network Guided Segment Anything Model for RGB-T Salient Object Detection | [Paper](https://arxiv.org/html/2504.05878v1) |
| 2025 | TCINet | TIM | Three-decoder Cross-modal Interaction Network for Unregistered RGB-T Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10947101) • [Code](https://github.com/zqiuqiu235/TCINet) |
| 2024 | ConTriNet | TPAMI | Divide-and-Conquer: Modality-aware Triple-Decoder Network for Robust RGB-T Salient Object Detection | [Paper](https://cser-tang-hao.github.io/contrinet.html) • [Code](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD) |
| 2024 | MSEDNet | NN | MSEDNet: Multi-scale fusion and edge-supervised network for RGB-T salient object detection | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023007384) • [Code](https://github.com/Zhou-wy/MSEDNet) |
| 2024 | PCNet | arXiv | Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network | [Paper](https://arxiv.org/abs/2412.14576) • [Code](https://github.com/Angknpng/PCNet) |

### Key 2023 Publications

| Model | Venue | Title | Resources |
|:-:|:-:|:--|:-:|
| LSNet | TIP | LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images | [Paper](https://ieeexplore.ieee.org/document/10042233) • [Code](https://github.com/zyrant/LSNet) |
| PRLNet | TIP | Position-Aware Relation Learning for RGB-Thermal Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10113883) • [Code](https://github.com/VisionVerse/PRLNet) |
| CAVER | TIP | CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10015667) • [Code](https://github.com/lartpang/caver) |
| WaveNet | TIP | WaveNet: Wavelet Network With Knowledge Distillation for RGB-T Salient Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10127616) • [Code](https://github.com/nowander/WaveNet) |

### Survey Papers

| Year | Venue | Title | Resources |
|:-:|:-:|:--|:-:|
| 2023 | 数据采集与处理 | Deep Learning Based Salient Object Detection: A Survey (Chinese) | [Paper](https://sjcj.nuaa.edu.cn/sjcjycl/article/pdf/202301002) |
| 2019 | CVM | 显著物体检测综述 (Chinese Translation of CVM 2019) | [Paper](https://mmcheng.net/wp-content/uploads/2021/10/CVM19_SalientObjectSurvey-CN.pdf) |

### 📚 SOD Resources

**Core RGB-T SOD Datasets:**

- **VT821 Dataset**: [Paper](https://link.springer.com/content/pdf/10.1007%2F978-981-13-1702-6_36.pdf) • [Download](https://drive.google.com/file/d/0B4fH4G1f-jjNR3NtQUkwWjFFREk/view?resourcekey=0-Kgoo3x0YJW83oNSHm5-LEw)
- **VT1000 Dataset**: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744296) • [Download](https://drive.google.com/file/d/1NCPFNeiy1n6uY74L0FDInN27p6N_VCSd/view)
- **VT5000 Dataset**: [Paper](https://arxiv.org/pdf/2007.03262.pdf) • [Download](https://pan.baidu.com/s/196S1GcnI56Vn6fLO3oXb5Q) (Code: y9jj)
- **VT723 Dataset**: [Paper](https://arxiv.org/pdf/2207.03558.pdf) • [Download](https://drive.google.com/file/d/12gEUFG2yWi3uBTjLymQ3hjnDUHGcgADq/view)

**Evaluation Tools:**

- **Python**: [CPU Version](https://github.com/lartpang/PySODEvalToolkit) • [GPU Version](https://github.com/zyjwuyan/SOD_Evaluation_Metrics)
- **MATLAB**: [Weighted F](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) • [Standard](http://dpfan.net/d3netbenchmark/)

**Related Repositories:**

- [RGB-T SOD & Semantic Segmentation Summary](https://github.com/zyrant/Summary-of-RGB-T-Salient-Object-Detection-and-Semantic-segmentation)
- [RGBT Salient Object Detection](https://github.com/lz118/RGBT-Salient-Object-Detection)

---

## 3. RGB-T Vehicle Detection

**Latest Update:** 2025-06-26

| Year | Model | Venue | Title | Resources |
|:-:|:-:|:-:|:--|:-:|
| 2023 | CALNet | ACM MM | Multispectral Object Detection via Cross-Modal Conflict-Aware Learning | [Paper](https://dl.acm.org/doi/10.1145/3581783.3612651) |
| 2023 | LRAF-Net | TNNLS | LRAF-Net: Long-Range Attention Fusion Network for Visible–Infrared Object Detection | [Paper](https://ieeexplore.ieee.org/abstract/document/10144688) |
| 2023 | GF-Detection | RS | GF-Detection: Fusion with GAN of Infrared and Visible Images for Vehicle Detection at Nighttime | [Paper](https://www.mdpi.com/2072-4292/14/12/2771) |
| 2023 | CMAFF | PR | Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320322002679) |
| 2022 | TSRA | ECCV | Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690501.pdf) |
| 2022 | UA-CMDet | TCSVT | Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning | [Paper](https://ieeexplore.ieee.org/abstract/document/9759286) |
| 2022 | RISNet | RS | Improving RGB-Infrared Object Detection by Reducing Cross-Modality Redundancy | [Paper](https://www.mdpi.com/2072-4292/14/9/2020) |

### 📚 Vehicle Detection Resources

**Datasets:**

- **DroneVehicle**: Partially aligned dataset • [Download](https://github.com/VisDrone/DroneVehicle)
- **VEDAI**: Strictly aligned dataset • [Download](https://downloads.greyc.fr/vedai/)
- **Multispectral Datasets**: With segmentation annotations • [Download](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)

---

## 4. RGB-T Crowd Counting

**Latest Update:** 2025-07-21

| Year | Model | Venue | Title | Resources |
|:-:|:-:|:-:|:--|:-:|
| 2025 | MHKDF | TCSVT | A Mutual Head Knowledge Distillation Framework for Lightweight RGB-T Crowd Counting | [Paper](https://ieeexplore.ieee.org/abstract/document/11080047) • [Code](https://github.com/BaoYangCC/MHKDF) |
| 2025 | MISF-Net | TMM | MISF-Net: Modality-Invariant and -Specific Fusion Network for RGB-T Crowd Counting | [Paper](https://ieeexplore.ieee.org/abstract/document/10855542) • [Code](https://github.com/QSBAOYANGMU/MISF-Net) |
| 2022 | MAFNet | arXiv | MAFNet: A Multi-Attention Fusion Network for RGB-T Crowd Counting | [Paper](https://arxiv.org/pdf/2208.06761.pdf) |
| 2022 | MAT | ICME | Multimodal Crowd Counting with Mutual Attention Transformers | [Paper](https://ieeexplore.ieee.org/abstract/document/9859777) |
| 2021 | IADM | CVPR | Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting | [Paper](https://arxiv.org/pdf/2012.04529.pdf) • [Code](https://github.com/chen-judge/RGBTCrowdCounting) |
| 2020 | MMCCN | ACCV | RGB-T Crowd Counting from Drone: A Benchmark and MMCCN Network | [Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Peng_RGB-T_Crowd_Counting_from_Drone_A_Benchmark_and_MMCCN_Network_ACCV_2020_paper.pdf) • [Code](https://github.com/VisDrone/DroneRGBT) |

### 📚 Crowd Counting Resources

**Datasets:**

- **RGBT-CC**: [Download](http://lingboliu.com/RGBT_Crowd_Counting.html)
- **DroneCrowd**: [Download](https://github.com/VisDrone/DroneCrowd)

**Evaluation Tools:**

- [DEFNet](https://github.com/panyi95/DEFNet)
- [BL+IADM](http://lingboliu.com/RGBT_Crowd_Counting.html)

---

## 5. RGB-T Pedestrian Detection

**Latest Update:** 2025-06-27

| Year | Model | Venue | Title | Resources |
|:-:|:-:|:-:|:--|:-:|
| 2024 | TFDet | TNNLS | TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection | [Paper](https://ieeexplore.ieee.org/document/10645696) • [Code](https://github.com/XueZ-phd/TFDet) |

*This section is actively expanding with new publications.*

---

## 6. RGB-T Semantic Segmentation

*This section will be updated with the latest semantic segmentation research.*

---

## 7. RGB-T Tracking

*This section will be updated with the latest tracking research.*

---

## 8. RGB-T Person Re-Identification

*This section will be updated with the latest re-identification research.*

---

## 9. RGB-T Image Alignment

*This section will be updated with the latest alignment research.*

---

## 🤝 Contributing

We welcome contributions to this collection! Please feel free to:

- 📝 **Submit new papers** via pull requests
- 🐛 **Report issues** or broken links
- 💡 **Suggest improvements** to the organization
- ⭐ **Star the repository** if you find it useful

### Contribution Guidelines

1. **Paper Format**: Include title, authors, venue, year, and links to paper/code
2. **Quality**: Only include papers from reputable conferences/journals
3. **Relevance**: Ensure papers are directly related to RGB-T research
4. **Completeness**: Provide as much information as possible (abstract, code, dataset)

---

## 📄 License

This collection is available under the [MIT License](LICENSE).

---

## 📬 Contact

For questions, suggestions, or collaborations, please open an issue or contact the maintainers.

---

**[⬆ Back to Top](#awesome-rgb-t-research-collection)**
