CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

Constance Wang
  * [LinkedIn](https://www.linkedin.com/in/conswang/)

Tested on AORUS 15P XD laptop with specs:  
- Windows 11 22000.856  
- 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz  
- NVIDIA GeForce RTX 3070 Laptop GPU  

This is an implementation of the [Edge-Avoiding À-Trous Wavelet Transform for fast Global
Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf) in CUDA, integrated into a CUDA pathtracer.

### Features
- Gbuffer to store normals and positions of each pixel
- Denoising pass that blurs pixels using an A-trous kernel, but avoids edges based on neighbouring pixels' ray-traced colour, normal, and position
- Path-tracer integration: the bonus features and performance testing for this assignment were done in base code of this project. However, I also integrated the denoiser into my project 3 pathtracer to visually test more complex scenes, see the [proj-4-denoiser](https://github.com/conswang/Project3-CUDA-Path-Tracer/pull/1) branch.

Showing the gbuffers as colours (click "Show G-buffers" with `SHOW_GBUFFER_NORMALS` or `SHOW_GBUFFER_POS` macros set to 1).
| Normals | Positions |
| ---- | ----|
| ![](img/box/normal-gbuffer.png) | ![](img/box/pos-gbuffer.png) |

### Performance

