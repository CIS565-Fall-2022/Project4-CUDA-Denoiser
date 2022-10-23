CUDA Denoiser
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Di Lu
  * [LinkedIn](https://www.linkedin.com/in/di-lu-0503251a2/)
  * [personal website](https://www.dluisnothere.com/)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

One major problem with Path tracing is that the completed image can often be noisy or grainy. This issue is amplified with scenes where rays are unlikely to hit the light: scenes with small lights, with a lot of geometry, etc. On top of that, the number of path trace iterations reaches a point of diminishing returns. The difference betweeen 500 and 1000 samples per pixel is not always obvious. 

In this project, I implemented an Edge-avoiding A-Trous Wavelet Transform filter for noisy path-traced images. The input to this function is a complete path-traced image, and the output is a denoised version of the image where large blocks of the same color will see less black specks. 

| Original 1 Iteration   | Denoised 1 Iteration |
| ----------- | ----------- |  ----------- |
| ![](img/nodenoise1.png)      |   ![](img/denoise1.png)     |
| Original 100 Iteration   | Denoise 100 Iterations | 
| ----------- | ----------- |  ----------- |
| ![](img/nodenoise100.png)      |   ![](img/denoise100.png)  |

## Core Features

## Performance Analysis

The point of denoising is to reduce the number of samples-per-pixel/pathtracing iterations needed to achieve an acceptably smooth image. You should provide analysis and charts for the following:

- how much time denoising adds to your renders
- how denoising influences the number of iterations needed to get an "acceptably smooth" result
- how denoising at different resolutions impacts runtime
- how varying filter sizes affect performance

In addition to the above, you should also analyze your denoiser on a qualitative level:

- how visual results vary with filter size -- does the visual quality scale uniformly with filter size?
- how effective/ineffective is this method with different material types
- how do results compare across different scenes - for example, between `cornell.txt` and `cornell_ceiling_light.txt`. Does one scene produce better denoised results? Why or why not?

Note that "acceptably smooth" is somewhat subjective - we will leave the means for image comparison up to you, but image diffing tools may be a good place to start, and can help visually convey differences between two images.

## Bloopers! :)

