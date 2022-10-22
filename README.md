CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Hanlin Sun
   * [LinkedIn](https://www.linkedin.com/in/hanlin-sun-7162941a5/)
* Tested on: Windows 10, i7-8750H @ 2.82GHz 32GB, NVIDIA Quadro P3200 

## Background

This project implements a **CUDA denoiser** based on the following paper: ["Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering,"](https://jo.dreggn.org/home/2010_atrous.pdf).

## Result

### Simple Blur

The A-Trous Wavelet filter is first implemented without any data weighting to achieve a simple blur effect. The result is compared to a Gaussian blur filter.
There are some artifacts in this implementation most noticeable around the edges of the image. However, for the most part the blur effect is achieved properly and indicates that the wavelet kernel is working well.

Original                   |  Gaussian                 
:-------------------------:|:-------------------------:
![](img/unDenoise.JPG)     |  ![](img/simpleBlur.JPG)  

### G-Buffer

For edge-avoiding weights, the normal, depth and position data per-pixel are stored in the G-buffer.

Position                   |  Normal                   |     Depth          |
:-------------------------:|:-------------------------:|:------------------:|
![](img/Position.JPG)     |  ![](img/NormalBuffer.JPG)      |![](img/Depth.JPG)

### Blur with Edge-Avoiding (A-trous method)

Denoising is achieved by adding the effect of weights in the convolution.
The parameters are tuned to produce a desirably smooth output. 

Original            |  Simple Blur                     | Blur with Edge-Avoiding (Final Result)
:-------------------------:|:-------------------------:|:-----------:
![](img/unDenoise.JPG)   |  ![](img/simpleBlur.JPG)      |  ![](img/denoise.JPG)

### Depth Reconstruct World Space Position

To achieve this, I first convert the ray intersection point(in world space) to the NDC space, and then remap the Z from [-1,1] to [0,1].
Then based on the depth buffer and corresponding ray intersection pixel position, successfully recompute the position value.

Without Depth Reconstruction   | With Depth Reconstruction  | 
:-------------------------:|:-------------------------:|
![](img/NoGBufferConstruct.JPG)     |  ![](img/WithDepth.JPG)  |


## Visual Analysis
### How visual results vary with filter size -- does the visual quality scale uniformly with filter size ?

From the images below, we can find out that the visual quality improves with increasing filter size.
However, they do not scale uniformly. There is a noticeable difference from 5x5 to 30x30. However, the difference is less significant from 30x30 to 60x60, and barely noticeable from 60x60 to 90x90, so it's not a linear improvement. 
Render iterations: 10

5x5    |30x30                      |  60x60                    | 90x90
:-----:|:-------------------------:|:-------------------------:|:-----------:
![](img/5x5.JPG)|![](img/30x30.JPG)|  ![](img/60x60.JPG)       | ![](img/90x90.JPG)

### How effective/ineffective is this method with different material types ?

The method is effective for diffuse materials and less effective for reflective materials. As shown below, the denoised result for the diffuse scene is representable of the actual outcome. However, in the specular scene, there will have noticeable blurs in the reflected surface, and if the position weight is increased, this blur will become larger.

Material Type | Original           |  Denoised  (Low Weight)   |  Denoised (High Weight)              
:------------:|:------------------:|:-------------------------:|:----------------------:|
Diffuse       |![](img/diffuse_origin.JPG)    |  ![](img/diffuse_denoise_low.JPG)   | ![](img/diffuse_denoise_high.JPG)
Specular      |![](img/specular_origin.JPG)   |  ![](img/specular_denoised_low.JPG) | ![](img/specular_denoised_high.JPG)


### How do results compare across different scenes - for example, between `cornell.txt` and `cornell_ceiling_light.txt`. Does one scene produce better denoised results ? Why or why not ?

In general, the denoised result is dependent on how noisy the input image is. 
For the default Cornell scene with smaller light area, the path traced result at 10 iterations is still very noisy. As such, denoising does not output good results.
However, for the Cornell scene with ceiling light, the path tracer converges faster with larger light area and thus produce significantly less noisy image. Accordingly, the output of the denoiser is much better.


Scene  | Original (10 Iterations)                          |  Denoised                
:-----:|:------------------:|:-------------------------:|
Cornell                     |![](img/diffuse_origin.JPG)   |  ![](img/diffuse_denoise.JPG)                         
Cornell Ceiling Light       |![](img/specular_noise.JPG)   |  ![](img/specular_denoise.JPG)     

## Performance Analysis

### How much time denoising adds to your renders ?

Since the denoising kernel is executed once during the last iteration of path tracing, the additional time from denoising is independent of the number of iterations that is run. 

![](img/denoiseTimeChart.JPG)  

### How denoising influences the number of iterations needed to get an "acceptably smooth" result ?

The purpose of denoising is to achieve the same smoothness/visual quality in image with less render iterations. Using a rendered image at 2000 iterations as the ground truth, we can see that the original input at 10 iterations is very noisy, But after applying denoising at 10 iterations render result, we immediately remove most of the noise. There are noticeable differences around the edge and shadow areas of the scene, which is a known limitation in the original reference paper. For the purpose of this project, we only look at smooth areas such as the walls and the floor for quality comparison. After 450 iterations of the path tracer, we roughly see the same amount of noise on the floor compared to the denoised version. Because of this, we consider 450 iterations as the acceptably smooth result, and thus the denoising reduces the required iterations for this specific example by **97.7%**!

Type    |Reference (2000 Iterations)     |  10 Iterations (Input)    |  Denoised (Output)   | 450 Iterations (Acceptably Smooth)          
:------:|:------------------:|:-------------------------:|:------------------:|:-------------------:
Image   |![](img/groundTruth.JPG)  |  ![](img/diffuse_origin.JPG)    | ![](img/diffuse_denoise_high.JPG) | ![](img/diffuse_450.JPG)

### How denoising at different resolutions impacts runtime ?

The denoising time increases proportionally with increasing image resolution. 
From 800x800 to 1200x1200, there are 2.25x more pixels mapping to 88.5% increase in time.
From 1200x1200 to 1600x1600, there are 1.78x more pixels mapping to 75.7% increase in time.
From 1600x1600 to 2000x2000, there are 1.57x more pixels mapping to 49.16% increase in time.

Render iterations: 20

![](img/resolution.JPG)  

### How varying filter sizes affect performance ?
The denoising time increases with increasing filter size. With increasing filter size, more passes/iterations are required to expand the 5x5 B3-spline kernel to cover the filter/blur size.

Render iterations: 20

![](img/fliterSizeChart.JPG)

### How Depth Reconstruct affect performance ?
Using Depth to recompute world space position bring a lot of performance increase. Since it's no longer need to read from another buffer and compute its value instead, save the time from reading position buffer and also the space for saving position buffer.

Render iterations: 20

![](img/DepthOptimize.JPG)