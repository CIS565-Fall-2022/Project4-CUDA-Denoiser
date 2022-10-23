CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Yu-Chia Shen
  * [LinkedIn](https://www.linkedin.com/in/ycshen0831/)
* Tested on: Windows 10, i5-11400F @ 4.3GHz 16GB, GTX 3060 12GB (personal)

# Overview
This project is to build a denoiser using colors and 2 geometry buffer: normals, and positions. The effect of the denoiser can smooth the image while not blur the edges between objects. The technique is based on the paper "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering". You can find the paper here: https://jo.dreggn.org/home/2010_atrous.pdf

![](./img/Laterns.png)

# Result

## **Before & After Comparison**
| Raw Image | After Denoise | 
| :--------------------------: | :-------------------: | 
| ![](./img/before.png) | ![](./img/after.png) |

| Blur Image | After Denoise | 
| :--------------------------: | :-------------------: | 
| ![](./img/beforeBlur.jpg) | ![](./img/after.png) |

## **GBuffer**
| Normal Buffer |  Position Buffer | 
| :--------------------------: | :-------------------: | 
| ![](./img/normals_img.png) | ![](./img/positions_img.png) |

# Performance Analysis

## Time Analysis for Denoiser

The following chart shows that the Denoiser time is approximately the same as the ray tracer in one iteration. Also, since denoise only needed to be applied in the final iteration, it doesn't not affect the whole render too much. It only count as one iteration of the ray tracer. 

![](./img/Resolution%20(pixel)%20vs%20Run%20Time%20(ms)%20vs%20iter.png)

## Visual Effect Analysis

### Visual Effect
| Iteration 1000 without Denoiser | Iteration 6000 without Denoiser | Denoised Image 100 iteration
| :--------------------------: | :-------------------: | :-------------------: | 
| ![](./img/iteration/a1000.png) | ![](./img/iteration/a6000.png)  | ![](./img/iteration/aResult.png) | 

We can see that the image needed 6000 iteration to achieve the effect of the denoiser. Even go throught 1000 iteration, the image is still noisy when zoom in. Therefore, the denoiser can greatly improve the visual effect with very less iteration.

### Image Difference Using tools
| Iteration 500 without Denoiser | Difference | Denoised Image 100 iteration 
| :--------------------------: | :-------------------: | :-------------------: | 
| ![](./img/iteration/cornell.2022-10-22_23-16-51z.500.0samp.png) | ![](./img/vs/vs500.png)  | ![](./img/denoiseResult.png) | 

| Iteration 1000 without Denoiser | Difference | Denoised Image 100 iteration 
| :--------------------------: | :-------------------: | :-------------------: | 
| ![](./img/iteration/cornell.2022-10-22_19-24-22z.1000.0samp.png) | ![](./img/vs/vs1000.png)  | ![](./img/denoiseResult.png) | 

| Iteration 6000 without Denoiser | Difference | Denoised Image 100 iteration 
| :--------------------------: | :-------------------: | :-------------------: | 
| ![](./img/iteration/cornell.2022-10-22_19-24-22z.6000.5samp.png) | ![](./img/vs/vs6000.png)  | ![](./img/denoiseResult.png) | 

## **Resolution vs Execute Time**
![](./img/chart/Resolution%20(pixel)%20vs%20Run%20Time%20(ms).png)

## **Filter Size vs Execute Time**
![](./img/chart/Filter%20Size%20(pixel)%20vs%20Run%20Time%20(ms).png)

## **Visual Effect with different Iteration**
| Denoised Image 5 iteration| Denoised Image 10 iteration | Denoised Image 20 iteration 
| :--------------------------: | :-------------------: | :-------------------: | 
| ![](./img/denoiseIter/cornell.2022-10-22_23-28-04z.5.5samp.png) | ![](./img/denoiseIter/cornell.2022-10-22_23-28-04z.10.4samp.png)  | ![](./img/denoiseIter/cornell.2022-10-22_23-28-04z.20.3samp.png) | 

| Denoised Image 40 iteration| Denoised Image 80 iteration | Denoised Image 100 iteration 
| :--------------------------: | :-------------------: | :-------------------: | 
| ![](./img/denoiseIter/cornell.2022-10-22_23-28-04z.40.2samp.png) | ![](./img/denoiseIter/cornell.2022-10-22_23-28-04z.80.1samp.png)  | ![](./img/denoiseIter/cornell.2022-10-22_23-28-04z.100.0samp.png) | 

We can see that the visual quality improve when we use more iterations. However, there is a limit for high iteration. The images with 80 iteration and 100 iteration seems the same.

## **Visual Effect with different Filter Size**
| 8 x 8 | 20 x 20 |  40 x 40 |
| :--------------------------: | :-------------------: |  :-------------------: | 
| ![](./img/visual_filter_size/f8.png)  | ![](./img/visual_filter_size/f20.png) | ![](./img/visual_filter_size/f40.png) |

| 80 x 80 | 200 x 200 | 400 x 400|
| :--------------------------: | :-------------------: |  :-------------------: | 
| ![](./img/visual_filter_size/f80.png)  | ![](./img/visual_filter_size/f200.png) | ![](./img/visual_filter_size/f400.png) |

You can see that the visual quality improve while increasing the filter size. The surface of the objects are smoother when filter size is larger. However, there is a limit for the filter size. When the filter size is larger than 80 x 80, the visual improvement is no longer exist. That means the visual quality does not scale uniformly with the filter size.

## Visual Effect with Different Material

| Material Type | Before | After |
| :--------------------------: | :-------------------: |  :-------------------: | 
| Diffuse  | ![](./img/material/diffuse.png) | ![](./img/material/diffuseDenoise.png) |
| Specular  | ![](./img/before.png) | ![](./img/after.png) |

We can see that the denoise effect is better on diffuse object than specular objects. The contour of the specular is blured after denoise.

## Visual Effect with Different Scenes
| Light Source | 100 iteration | 1000 iteration |
| :--------------------------: | :-------------------: |  :-------------------: | 
| Small Light Source  | ![](./img/NewScene/cornell.2022-10-22_23-49-30z.101.1samp.png) | ![](./img/NewScene/cornell.2022-10-22_23-49-30z.1026.3samp.png) |
| Large Light Source  | ![](./img/after.png) | ![](./img/NewScene/Large.png) |

We can see that for 100 iteration, large light source has better denoised result. This is because before denoise, small light source has a more noisy image. Therefore, too many noise in the image result in a poor denoiser effect. However, when the image converge after 1000 iteration, the denoiser effect become better.
