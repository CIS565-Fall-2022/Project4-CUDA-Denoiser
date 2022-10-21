CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zhangkaiwen Chu
  * [LinkedIn](https://www.linkedin.com/in/zhangkaiwen-chu-b53060225/)
* Tested on: Windows 10, R7-5800H @ 3.20GHz 16GB, RTX 3070 Laptop GPU 16310MB (Personal Laptop)

This project implement a pathtracing denoiser that uses geometry buffers to guide a smoothing filter, which is based on the paper ["Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering"](https://jo.dreggn.org/home/2010_atrous.pdf). This paper use wavelet to approximmate Gaussion filter. By guiding the filter with edge-stopping function, the denoiser will not make the whole image blurry.

| raw pathtraced image | simple blur | blur guided by G-buffers |
|---|---|---|
|![](img/origin.png)|![](img/blur.png)|![](img/denoise.png)|

The G-buffer contains position and surface normal informations.
| scene | position | surface normal |
|---|---|---|
|![](img/scene.png)|![](img/position.png)|![](img/normal.png)|
