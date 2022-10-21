CUDA Denoiser For CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Haoquan Liang
  * [LinkedIn](https://www.linkedin.com/in/leohaoquanliang/)
* Tested on: Windows 10, Ryzen 7 5800X 8 Core 3.80 GHz, NVIDIA GeForce RTX 3080 Ti 12 GB

# Overview
This project is a CUDA-based pathtracing denoiser that uses geometry buffers (G-buffers) to guide a smoothing filter. It is based on the paper "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering" and it helps produce a smoother appearance in a pathtraced image with fewer samples-per-pixel. 

Denoiser Off | Denoiser On
:----------:|:-----------:
![](img/Denoiser/denoise-off.png) | ![](img/Denoiser/denoise-on.png) 

# Table of Contents  
* [Features](#features)   
* [Performance Analysis](#performance)   
* [Reference](#reference)

# <a name="features"> Features</a>
### Core features
* G-Buffer Visualization
* A-Trous Filtering
* Edge-Avoiding Filtering
### Additional features
* Gaussian Filtering


# <a name="performance">Performance Analysis</a>
* How much time denoising adds to the renders
* how denoising influences the number of iterations needed to get an "acceptably smooth" result
* how denoising at different resolutions impacts runtime
* how varying filter sizes affect performance
* how visual results vary with filter size -- does the visual quality scale uniformly with filter size?
* how effective/ineffective is this method with different material types
* how do results compare across different scenes - for example, between `cornell.txt` and `cornell_ceiling_light.txt`. Does one scene produce better denoised results? Why or why not?
* A-Trous vs. Gaussian Filtering



# <a name="reference">Refrence</a>
* [Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)
* [Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A)
* [A Survey of Efficient Representations for Independent Unit Vectors](http://jcgt.org/published/0003/02/01/paper.pdf)
* ocornut/imgui - https://github.com/ocornut/imgui
