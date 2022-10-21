CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* David Li
* [LinkedIn](https://www.linkedin.com/in/david-li-15b83817b/)
* Tested on: Windows 10, Intel(R) Core(TM) i9-10980HK CPU @ 2.40GHz 32Gb, GTX 2070 Super (Personal Computer)

[Repo Link](https://github.com/theCollegeBoardOfc/Project4-CUDA-Denoiser)

### Denoiser Motivation
Path tracers are excellent at generating realistically lit scenes, but can be quite slow to converge even when built on the GPU. Path tracers early in their iteration cycle produce images that appear staticky and rough. Denoisers act as a sort of more precise blur effect. For each pixel, the colors of its neighboors are gathered, and accumulated. The magnitude of color taken from each neighboor is dependent on the neighboors color, position in the scene and normal vector with respect to the original pixel. By doing so, we can achieve edge detection, choosing which neighboor pixels actually matter in generating a better color and render as a whole. [This paper](https://jo.dreggn.org/home/2010_atrous.pdf) outlines the implementation of Edge-Avoiding A-Trous Wavelet transformation (The Denoising Algorithm). 

### Examples:
<br />
Here is what a standard gaussian blur looks like. It sill uses the A-trous wavelet to approximate a gaussian kernel, but no normal, position or color weights are used for edge detection.

![](img/blur_1.PNG)
<br />

With just the position weights, the image develops a bit more crispness. Especially surrounding the sphere. The wall and sphere have drastically different position values lead to this effect.

![](img/p_w.PNG)
<br />
With just normal weights, the corners of the wall are more clear. The sphere still has the same level of crispness on the edges as well. Here all the crisp edges are a result of objects having different normals than other. Note though, that the sphere itself is quite blurry because all the pixels showing the sphere have different normals.

![](img/n_w.PNG)
<br />
With just color weights, there remains some garininess in the image, even though it has been through the same number of iterations as past images. This is because in the non denoised image, there remains a lot of graininess. Since the color weights depend on neighbooring pixels to be similar color to the original pixel, high amounts of graininess in the original image will cause the denoiser to discard neighboors of disimilar values, even if their addition would make the final render appear better.

![](img/c_w.PNG)
<br />

Lastly, here is the render with all weights applied, and with no denoising at all.
![](img/all.PNG)
![](img/none.PNG)
<br />

### Performance Analysis

Denoising is actually a very efficient process. For starters it's also a highly parallelizable process, just like path tracing, meaning it can be accelerated with use of the GPU. It also only needs to be done once at the final iteration of the pathtracer. Even if the pathtracer has a relatively modest 50 iterations, the addition of a denoiser at the end would have little impact on the runtime. As a result, we will be denoising at every iteration, to get a better idea of the added runtime of using the denoiser at one iteration. Unless otherwise specified, filtersize will be 80 and image resolution is 800 x 800. 

<br />
It's difficult to quantify how much quicker the denoising causes convergence. Here is a scene from before of the denoised image at 10 iterations.

![](img/all.PNG)

<br />
And here is the same scene but at 200 iterations but with no denoising. 

![](img/200n.PNG)

<br />
Both images are lacking in different ways. The denoised image is splotchy as not enough convergence has occured for the blur effect to generate uniform colors. The nondenoised image is simply still a bit grainy. Both could be considered acceptably converged, as nothing is off at first glance though. 

<br />
Also, here is the scene at 200 iterations while denoised, and 2000 iterations not denoised. 

![](img/200d.PNG) 

![](img/2000n.PNG) 

<br />
The scenes are pretty indistinguishible. It may be safe to say that the denoiser, when paramaters are adjusted properly, can reduce the number of required iterations for an image to converge by a factor of 10 to 20. However, this statement is completely subjective.
