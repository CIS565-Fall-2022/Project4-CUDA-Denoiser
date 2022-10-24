CUDA Denoiser
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Di Lu
  * [LinkedIn](https://www.linkedin.com/in/di-lu-0503251a2/)
  * [personal website](https://www.dluisnothere.com/)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

One major problem with Path tracing is that the completed image can often be noisy or grainy. This issue is amplified with scenes where rays are unlikely to hit the light: scenes with small lights, with a lot of geometry, etc. On top of that, the number of path trace iterations reaches a point of diminishing returns. The difference betweeen 500 and 1000 samples per pixel is not always obvious. 

How might we reduce the number of samples-per-pixel/pathtracing iterations needed to obtain an acceptably smooth image? One method is to use a denoising algorithm.

In this project, I implemented an Edge-avoiding A-Trous Wavelet Transform filter for noisy path-traced images. The input to this function is a complete path-traced image, and the output is a denoised version of the image where large blocks of the same color will see less black specks. The following is a sample of the output when there is only 1 iteration on Pathtrace vs. when there are 100 Iterations. It can be seen that the denoised version of the first set of images is still noisy compared to the denoised version of a more converged image.

| Original 1 Iteration   | Denoised 1 Iteration |
| ----------- | ----------- | 
| ![](img/nodenoise1.png)  |   ![](img/denoise1.png)   |

| Original 100 Iteration | Denoise 100 Iterations | 
| ----------- | ----------- |
| ![](img/nodenoise100.png)  |  ![](img/denoise100.png)  |

For the above images, I used the following settings:
- Filter Size: 80
- Color Weight: 200
- Normal Weight: 0.05
- Position Weight: 4
- Kernel Size: 5x5 

## Core Features

#### Simple Blurring

The first step in implementing Edge Avoiding ATrous Denoising is to average each pixel's neighboring color without taking any weights into consideration. (This is very similar to implementing Gaussian blur, except there is no kernel of weights). We simply divide the sum of all neighboring colors by the number of neighbors. In my case, I use a 5x5 area around the pixel as its neighbors.

| Original | Simple Blurring | Photo Editing Software Blur
| ----------- | ----------- | ----------- |
| ![](img/nodenoise100.png)  |  ![](img/simpleBlur.png)  | ![](img/pixlrBlur.png)

#### Geometry Buffers: Position, Normal, and Time To Intersect

For debugging purposes, we also have to implement a GBuffer which contains information about each pixel that we will use to calculate edge-avoidance later on. For the purposes of this project, I created a GBuffer struct that stores glm::vec3 for position and normal, and a float representing the time it took for the intersection to happen on that pixel.

| Position | Normal | Time to Intersect
| ----------- | ----------- | ----------- |
| ![](img/position.png)  |  ![](img/normal.png)  | ![](img/timeToIntersect.png)

#### Edge-Avoiding ATrous Denoising (As seen on a simple scene)

Now that we have information from blur and the GBuffer, I followed the implementation details suggested by the following paper: [Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf). 

Like Gaussian blur, we will use a kernel of weights to average out each neighboring pixel's contribution. However, we will also "expand" the kernel by applying these kernel values to gradually further and further apart neighbors, as shown in the diagram below:

![](img/atrousDescription.png)

I added a new function that would denoise the output image rather than show the original image. This function only gets called when the number of pathtrace calls has reached its limit. This function will take as an input the final rendered result of the scene. Each pixel will expand its kernel over a set size (filter size) to add and average these neighboring pixels based on weights from the kernel. 


| Original (50 Iterations) | Denoised (50 Iterations) |
| ----------- | ----------- |
| ![](img/noDenoise50.png)  |  ![](img/denoise50.png)  |

## Performance Analysis

- **how much time denoising adds to your renders**

If there are many iterations, then denoising should not add much time to the overall production time. This is because denoising only occurs once after the entire image has converged, and is thus not correlated to the number of iterations in the image. For example, with 1 iteration, perhaps the total amount of production time will increase by a lot. However, with something like 1000+ iterations, the amount of extra time it takes to denoise will hardly add much:

![](img/itersDenoise.png)

- **how denoising influences the number of iterations needed to get an "acceptably smooth" result**

| Acceptably Smooth Without Denoise (500 Iters) | Acceptably Smooth Denoised (50 Iterations) | No Denoise (50 Iterations) |
| ----------- | ----------- | ----------- |
| ![](img/acceptableSmooth700.png)  |  ![](img/acceptableSmooth50.png)  | ![](img/noDenoise50.png) 

| Acceptably Smooth Without Denoise (1000 Iters) | Acceptably Smooth Denoised (100 Iterations) | No Denoise (100 Iterations) 
| ----------- | ----------- | ----------- |
| ![](img/acceptableSmooth1000.png)  |  ![](img/acceptableSmooth100.png)  |  ![](img/nodenoise100.png) |


- **how denoising at different resolutions impacts runtime**


- **how varying filter sizes affect performance**



In addition to the above, you should also analyze your denoiser on a qualitative level:

- **how visual results vary with filter size -- does the visual quality scale uniformly with filter size?**

The visual quality does not scale uniform with filter size. Although the higher the filter is, the more smooth the image, You have to roughly double the size of the filter each time in order to see progressive improvement. This is to say that although there was a large difference between Filter size 20 and size 12, there's little to no change between filter 28 and filter 20. 

| Original | FilterSize = 12  | FilterSize = 20
| ----------- | ----------- | ----------- |
| ![](img/noDenoise.png)  |  ![](img/denoise12.png)  | ![](img/denoise20.png)

| FilterSize = 36 | FilterSize = 68  | FilterSize = 150
| ----------- | ----------- | ----------- |
| ![](img/denoise36.png)  |  ![](img/denoise68.png)  | ![](img/denoise150.png)

- **how effective/ineffective is this method with different material types**

My understanding and theory would be that this method works significantly better for diffuse surfaces than specular. Let's consider a specular cube. When we calculate the final color of a pixel in Edge-Avoiding ATrous, we take into account its position, color, and normal. However, on a particular face of a cube, the normals are all the same for each of those pixels. If a specular cube is reflecting the corner of a room, for example, then the reflection would be more blurry than the actual corner of the room, since the walls making up the corner of the room will have different normals.

However, for a diffuse object, these blurring effects will be much less obvious, unless the diffuse object has multiple colors.

For the following image, this difference between specular and diffuse is seen clearly:

![](img/denoise1.png)



- **how do results compare across different scenes - for example, between `cornell.txt` and `cornell_ceiling_light.txt`. Does one scene produce better denoised results? Why or why not?**

Generally speaking, the image with a larger light will produce better denoised results as seen below (however, this does not mean that larger light intensity will result in a more denoised image. In fact, the more the light intensity, often times the more noisy the resuling image becomes). For scenes with larger lights, more rays are likely to hit the light, which reduces the number of black specks on the screen. 

| Original Cornell Scene (50 Iterations, 3 Light Intensity) | Denoised Cornell Scene (50 Iterations, 3 Light Intensity) |
| ----------- | ----------- |
| ![](img/smallLightNoDenoise.png)  |  ![](img/smallLightDenoise.png)  |


| Large Light Cornell Scene (50 Iterations, 1 Light Intensity) | Denoised Large Light Cornell Scene (50 Iterations, 1 Light Intensity) |
| ----------- | ----------- |
| ![](img/largeLightNoDenoise.png)  |  ![](img/largeLightDenoise.png)  |

## Bloopers! :)

