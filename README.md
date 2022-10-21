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

After implemented the Gaussian Blur, I slightly change the code and make it into A-Trous Wavelet transform. The idea of A-Trous is to approximate gaussian by iteratively applying sparese blurs of increasing size. The 5*5 kernel is still used as the filter, but for each iteration, accrording to the given filter size, said 16 * 16, we will space out the samples according to the offset three times(iterations) and added the weighted result to create the blurry result.

//// three result ps, my gaussian, my a-trous

After testing the denoisor with my custom scene, I found the time is dramatically decreased. It took me hours to render the scene with 3739 iterations to create a not bad result. But it only took me about 3 minutes to render the scene with 39 iterations ot create a smooth image.

//// two custome scene

# Performance Analysis

how much time denoising adds to your renders

how denoising influences the number of iterations needed to get an "acceptably smooth" result

how denoising at different resolutions impacts runtime

how varying filter sizes affect performance


how visual results vary with filter size -- does the visual quality scale uniformly with filter size?

how effective/ineffective is this method with different material types

how do results compare across different scenes - for example, between cornell.txt and cornell_ceiling_light.txt. Does one scene produce better denoised results? Why or why not?




