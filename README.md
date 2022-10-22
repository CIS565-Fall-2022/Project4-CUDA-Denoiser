CUDA Denoiser For CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Wenqing Wang
  * [LinkedIn](https://www.linkedin.com/in/wenqingwang0910/) 
* Tested on: Windows 11, i7-11370H @ 3.30GHz 16.0 GB, GTX 3050 Ti

## Overview
This project implements a pathtracing denoiser that uses geometry buffers (G-buffers) to guide a smoothing filter. The impelmentation is based on the paper "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering," by Dammertz, Sewtz, Hanika, and Lensch. The paper could be found here: https://jo.dreggn.org/home/2010_atrous.pdf

| 300 iterations |300 iterations + denoiser |
|--|--|
|![cornell 2022-10-22_00-58-24z 300samp](https://user-images.githubusercontent.com/33616958/197310311-90423e9f-5a26-4884-bb23-013639b7eb4a.png)|![cornell 2022-10-22_00-32-41z 300samp](https://user-images.githubusercontent.com/33616958/197308568-a95d1f1e-2961-47fa-8d47-9519e7b07cf0.png)|

| 1500 iterations | 300 iterations + denoiser |
|--|--|
|![cornell 2022-10-22_01-04-47z 1500samp](https://user-images.githubusercontent.com/33616958/197310320-b51ccc73-aacb-4ac7-af51-4559ce170887.png)|![cornell 2022-10-22_00-32-41z 300samp](https://user-images.githubusercontent.com/33616958/197308568-a95d1f1e-2961-47fa-8d47-9519e7b07cf0.png)|

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

From the rendering above, we can see that with only A-tour implemented, the whole image is blurred out by our denoiser, which is not the denoising result we expect. To keep the denoised edges clear, we store the position, normal and albedo (color) information in the G-Buffer. In general, if the values of these attributes change dramatically between 2 pixels, we assume that an edge is present.  In the third figure, we can see that after implementing the G-Buffer, the edges of the sphere, the junction of the left and right walls with the floor and ceiling are not blurred anymore. For different scenes, we can also tune the weight for each properties to get a better denoised result.

### Small light source vs. large light source
|small light source | large light source|
|--|--|
|![cornell 2022-10-22_00-46-15z 100samp](https://user-images.githubusercontent.com/33616958/197309832-c7f31d90-70a7-4963-ba0b-9171f971b80b.png)|![cornell 2022-10-22_00-44-51z 100samp](https://user-images.githubusercontent.com/33616958/197309666-ff18f840-c93c-4d17-8736-a5d329362ee4.png) |

The amount of light in the scene is another factor that can affect the effectiveness of our denoiser. Both images above are rendered with the same filter size (=80), number of iterations (=100) and G-Buffer weights, and we can see that the denoiser can provide less noisy results in the second scene. I think this is because for the same number of iterations, our scene can converge faster with a larger light source because the light ray has a higher probability of hitting the light source.

## Performance Analysis

### Time vs. Resolution
![PA_time_vs_resolution](https://user-images.githubusercontent.com/33616958/197292266-36120899-5396-4cbf-b83b-f097fc451459.png)

We can see from the results in the above figure that the denoising time in general increases linearly with the increase of the scene resolution size. 

### Time vs. Filter size
![PA_time_vs_filter](https://user-images.githubusercontent.com/33616958/197292281-4d207eb5-2717-4a12-98b7-90b1993e010c.png)

We can see from the results above that the denoising time in general increases logarithmically with the increase in filter size.  

### Time vs. Iterations
![PA_time_vs_iter](https://user-images.githubusercontent.com/33616958/197292287-66688b4c-6b56-41bd-9e80-0c1448325591.png)

As we can see from the above results, while the overall iteration time of our path tracer increases linearly with the number of iterations, the running time of denoiser remains roughly constant. From the paper we can learn that the time complexity of the A-tour algorithm is depend on filter size and scene resolution (number of pixels), but not to the properties of the pixels themselves. Therefore, the result is in line with our expectation.

## Blooper

<img width="402" alt="blooper1" src="https://user-images.githubusercontent.com/33616958/197292408-e2df1aa0-dfd8-4097-b388-eea226175315.png">

<img width="596" alt="blooper2" src="https://user-images.githubusercontent.com/33616958/197292412-5168a47b-2a45-4bae-8ced-c95d22905184.png">


