CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Dongying Liu
* [LinkedIn](https://www.linkedin.com/in/dongying-liu/), [personal website](https://vivienliu1998.wixsite.com/portfolio)
* Tested on:  Windows 11, i7-11700 @ 2.50GHz, NVIDIA GeForce RTX 3060

# Project Description

For this project, I implemented an denoiser into my path-tracer based on the [Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)

Pathtraced result are often noisy or grainy. And the noise reduction usually doesn't scale linearly with more iterations. So, the goal for this project is to implement a denoisor to decrease the time to render a nice result. Bacially, the idea is to blur the image with the edge preserving. 

I started with implemented a simple Gaussian Blur. With a 5*5 kernel, for every pixel, I added the weighted result of the 24 neighbors of the it so the final picture is blurry.

After implemented the Gaussian Blur, I slightly change the code and make it into A-Trous Wavelet transform. The idea of A-Trous is to approximate gaussian by iteratively applying sparese blurs of increasing size. The 5x5 kernel is still used as the filter, but for each iteration, accrording to the given filter size, said 16x16, we will space out the samples according to the offset three times(iterations) and added the weighted result to create the blurry result.

Then, I added a edge preserving to the A-Trous Wavelet Transform. With the help of per-pixel color, normal and position data to preserve edges when blurring according to the paper. 

//// gbuffer
//// three result ps gaussian, my a-trous, my a-trous with edge preserving
|Photoshop Gaussian | My Gaussian | A-Trous | A-Trous with Edge Preserving |
|-------------|-------------|-------------|-------------|
|<img width="200" height="200" src="/img/gaussianPS.png">|<img width="200" height="200" src="/img/gaussian.jpg">|<img width="200" height="200" src="/img/atrous.jpg">|<img width="200" height="200" src="/img/filtersize80.jpg">|

After testing the denoisor with my custom scene, I found the time is dramatically decreased. It took me hours to render the scene with 3739 iterations to create a not bad result. But it only took me about 3 minutes to render the scene with 39 iterations ot create a smooth image.

| Original 3739 iterations| Denoised 39 iterations|
|-------------|-------------|
| <img width="400" height="400" src="/img/customScene_3739samp.png">|<img width="400" height="400" src="/img/customSceneDenoised.jpg">|

# Performance Analysis

## Time added by Denoiser
The chart shows the time added when denoiser is on. There's about 5-10ms added when using denoiser. 

<img width="600" src="/img/timeAddedIter.png">

## Iteration for smooth result decreased by denoiser
However, the smooth result denoiser added to the final image save a lot more time than using more iterations to create the smooth result.
The right image without denoise used 500 iterations. The left image with denoise only used 10 iterations.
| Original 500 iterations| Denoised 10 iterations|
|-------------|-------------|
| <img width="400" height="400" src="/img/500iterResultNoDonoise.jpg">|<img width="400" height="400" src="/img/10iterResultDenoise.jpg">|

## Denoiser under different resolutions
With filter size and kernel size remained the same and resolution of the image increased, as the chart shows, denoiser took more time to create a smooth result. This make sense because there are more pixels for denoiser to go over even thought the filter size and kernel size is not changing.

<img width="600" src="/img/timeResolution.png">

## Filter size and denoise time
With kernel size remained the same, as the chart shows, denoiser took more time to create a smooth result. This make sense because when filter size increase, the iteration of the A-Trous increase, it will take more times for more iterations.


<img width="600" src="/img/timeFilterSize.png">

## Filter size and denoise visual result
Because of the calculation of the iteration base on the filtersize, the visual quality does not scale uniformly with filter size. Below is the filter size I have been testing with. Some of the filter size actually have the same iteration, so there are no obvious different between them.

| Filtersize | Iterations | Result | Filtersize | Iterations | Result | 
|-| ------------- | -------- |-| ------------- | -------- |
| 16 | 1 | <img width="300" src="/img/filtersize16.jpg"> | 30 | 2 | <img width="300" src="/img/filtersize30.jpg">|
| 50 | 3 | <img width="300" src="/img/filtersize50.jpg"> | 80 | 4 | <img width="300" src="/img/filtersize80.jpg"> |
| 200 | 5 | <img width="300" src="/img/filtersize200.jpg"> |

## Denoiser and materials
In my custom scene, the deer is glass which is refractive and the back wall is mirror which is reflective, others are all diffuse material. And we can tell from it that denoiser works well with diffuse material because the surface is mostly one color. However, denoised result looks not good for reflective and refreactive material, since they all looks blurry after denoise. These two materials create reflection and refraction of the surrounding scene, the edge of the reflected/refracted object need to be clear to make the material credible. 
<p align="center">
  <img width="400" height="400" src="/img/customSceneDenoised.jpg">
</p>

## Denoiser and lighting conditions
With the same kernel size and filter size, the cornell.txt scene needed about 100 iterations to create a smooth result while the cornell_ceiling_light.txt only needed 10 iterations. cornell_ceiling_light.txt produce better denoised result. I think the reason is this secene has a bigger ligth, so more rays will end with the light and contribute to the final result, so the image will convert faster. When the origin image quality is better, I think the denoised result will be better.

| Large Light 10 iterations| Small Light 100 iterations|
|-------------|-------------|
| <img width="400" height="400" src="/img/lightcondition10.jpg">|<img width="400" height="400" src="/img/lightcondition100.jpg">|
