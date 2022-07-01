# Daily Logs

## Table of Contents

- [2022](#2022)  
    - [2022/07](#2022/07) 

## 2022
### 2022/07

- **2022/07/01, Friday.**

 1. <u>Multi-View Transformer for 3D Visual Grounding.</u> [[PDF]](https://arxiv.org/pdf/2204.02174.pdf) [[Code]](https://github.com/sega-hsj/MVT-3DVG)
    - Main Idea: Two modals: Point cloud and text. Learn a multi-modal representation independent from from its sepecific single view. Different rotation matrixes are used for robust multi-view representation.
    - Experiments: Nr3D:55.1%, Sr3D:58.5%, Sr3D+:59.5%(SOTA) ScanRefer:40.80%(GOOD)
    - Reproduce Notes: 
        * 1 RTX 3090 takes almost 15h for Nr3D.
        * Replacing all mentions of AT_CHECK with TORCH_CHECK in ./referit3d/external_tools/pointnet2/_ext_src/src.
        * Point Cloud Visualization tool: open3d [[Package]](https://github.com/isl-org/Open3D)

    <p align="center"> <img src='imgs/2022/07/20220701_MVT_3DVG.png' align="center" height="200px"> </p>