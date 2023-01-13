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
- Parameters: filterSize = 5 * 2^(# of iterations of A-trous), and colorWeight, normalWeight, positionWeight which correspond to the sigma parameter in the weight calculations from the paper for colors, normals, and positions respectively
- Path-tracer integration: the bonus features and performance testing for this assignment were done in base code of this project. However, I also integrated the denoiser into my project 3 pathtracer to visually test more complex scenes, see the [proj-4-denoiser](https://github.com/conswang/Project3-CUDA-Path-Tracer/pull/1) branch.

Showing the gbuffers as colours for cornell ceiling light scene (click "Show G-buffers" with `SHOW_GBUFFER_NORMALS` or `SHOW_GBUFFER_POS` macros set to 1).
| Normals | Positions |
| ---- | ----|
| ![](img/box/normal-gbuffer.png) | ![](img/box/pos-gbuffer.png) |

Denoiser on cornell ceiling light scene (10 samples) with filterSize = 80, colorWeight = 1.804, normalWeight = 0.309, positionWeight = 7.113. The a-trous only image shows the blur effect of the A-trous kernel only, without edge detection.

|Original | A-trous only | Denoised with edge detection |
| --- | ---| ---|
| ![](img/box/orig.png) | ![](img/box/a-trous-only.png) | ![](img/box/denoised.png) |

Denoiser tested on complex scene: [motorcycle.gltf](https://github.com/conswang/Project3-CUDA-Path-Tracer/blob/main/scenes/motorcycle/motorcycle.gltf) with filterSize = 320, colorWeight = 4, normalWeight = 1, positionWeight = 1.

| Samples | Original | Denoised |
|-----| ----- | ---- |
| 10 | ![](img/motorcycle/10-samples-noisy.png) | ![](img/motorcycle/10-samples-denoised.png) |
| 20 | ![](img/motorcycle/20-samples-noisy.png) | ![](img/motorcycle/20-samples-denoised.png) |
| 50 | ![](img/motorcycle/50-samples-noisy.png) | ![](img/motorcycle/50-samples-denoised.png)
| 100 | ![](img/motorcycle/100-samples-noisy.png) | ![](img/motorcycle/100-samples-denoised.png)

For this scene, it takes about 100 iterations to get a smooth result. Note that it's very hard to save the details on surfaces like the vending machine. This is because the normals of neighbouring pixels are very similar (the surface is almost flat), positions are similar (object is centered and directly faces the camera), and the colours are similar, so the overall blend weight is high. To preserve the edges on the coke bottle, we'd need sigma values so small that the denoising effect in other areas would be greatly reduced. In other words, a big drawback of edge-avoiding A-trous is that different objects would render better with different parameters, but we use a uniform filter size and weights across the image.

### Denoising Render Time

Denoising should have a very small effect on render time, since it runs in constant time in parallel on the GPU. We only need to generate the g buffers on the first bounce of the first iteration. Then, we only need to denoise once after raytracing is complete. Both steps launch kernels that run in constant time for each pixel in parallel.

#### Performance Measurement

When the `MEASURE_DENOISE_PERF` flag is set to 0, each iteration is denoised for more convenient debugging.  When set to 1, only the last path-traced iteration is denoised for a more accurate performance analysis. I measured the total path-tracing time (from `pathtraceInit`, up to but not including `pathtraceFree`), the g-buffer initialization time, and the denoising time.
