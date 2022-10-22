CUDA Denoiser For CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Wenqing Wang
  * [LinkedIn](https://www.linkedin.com/in/wenqingwang0910/) 
* Tested on: Windows 11, i7-11370H @ 3.30GHz 16.0 GB, GTX 3050 Ti

## Overview
This project implements a pathtracing denoiser that uses geometry buffers (G-buffers) to guide a smoothing filter. The impelmentation is based on the paper "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering," by Dammertz, Sewtz, Hanika, and Lensch. The paper could be found here: https://jo.dreggn.org/home/2010_atrous.pdf

| 2500 iterations | 300 iterations + denoiser |
|--|--|
|![main_with_des](https://user-images.githubusercontent.com/33616958/194778938-be4f9d29-40d6-491c-b5fc-df4efbe4aff5.jpg)|![cornell 2022-10-22_00-32-41z 300samp](https://user-images.githubusercontent.com/33616958/197308568-a95d1f1e-2961-47fa-8d47-9519e7b07cf0.png)|

## Features
### G-Buffers visualization
| denoised raw image | per-pixel positions (scaled down) | per-pixel normals|
|---|---|---|
|<img width="362" alt="normal_celling" src="https://user-images.githubusercontent.com/33616958/197309468-495c922a-97d3-49c1-b130-9dc9c4168711.png">|<img width="361" alt="gbuff_pos" src="https://user-images.githubusercontent.com/33616958/197309460-b139e7e1-f16c-463b-8388-c006e0eb6ee6.png">|<img width="359" alt="gbuff_nor" src="https://user-images.githubusercontent.com/33616958/197309459-3732457c-0f7d-45b2-8a81-f7f08aa563af.png">|

### A-trous w/wo G-buffer(edge voiding)
The following renders are base on 100 iterations.
| raw image | raw image + A-tour | raw image + A-tour + G-Buffer|
|---|---|---|
|![cornell 2022-10-22_00-44-51z 100samp](https://user-images.githubusercontent.com/33616958/197309666-ff18f840-c93c-4d17-8736-a5d329362ee4.png) | ![cornell 2022-10-22_00-44-28z 100samp](https://user-images.githubusercontent.com/33616958/197309679-eed15a11-d924-4a66-825e-5bba3216efe4.png) | ![cornell 2022-10-22_00-42-13z 100samp](https://user-images.githubusercontent.com/33616958/197309695-1c00f6d8-7af4-49ab-a1c6-c8adb8500a62.png) |

### Small light source vs. large light source
The following renders are base on 100 iterations.
|small light source | large light source|
|--|--|
| ![cornell 2022-10-22_00-46-15z 100samp](https://user-images.githubusercontent.com/33616958/197309832-c7f31d90-70a7-4963-ba0b-9171f971b80b.png) |![cornell 2022-10-22_00-44-51z 100samp](https://user-images.githubusercontent.com/33616958/197309666-ff18f840-c93c-4d17-8736-a5d329362ee4.png) |

## Performance Analysis

### Time vs. Resolution
![PA_time_vs_resolution](https://user-images.githubusercontent.com/33616958/197292266-36120899-5396-4cbf-b83b-f097fc451459.png)


### Time vs. Filter size
![PA_time_vs_filter](https://user-images.githubusercontent.com/33616958/197292281-4d207eb5-2717-4a12-98b7-90b1993e010c.png)


### Time vs. Iterations
![PA_time_vs_iter](https://user-images.githubusercontent.com/33616958/197292287-66688b4c-6b56-41bd-9e80-0c1448325591.png)


## Blooper

<img width="402" alt="blooper1" src="https://user-images.githubusercontent.com/33616958/197292408-e2df1aa0-dfd8-4097-b388-eea226175315.png">

<img width="596" alt="blooper2" src="https://user-images.githubusercontent.com/33616958/197292412-5168a47b-2a45-4bae-8ced-c95d22905184.png">


