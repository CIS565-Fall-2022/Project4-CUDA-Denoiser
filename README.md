CUDA Path Tracer Denoiser
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**


* Yilin Liu
  * [LinkedIn](https://www.linkedin.com/in/yilin-liu-9538ba1a5/)
  * [Personal website](https://www.yilin.games)
* Tested on personal laptop:
  - Windows 10, Intel(R) Core(TM), i7-10750H CPU @ 2.60GHz 2.59 GHz, RTX 2070 Max-Q 8GB

Overview
=============

In this project I implemented a pathtracing denoiser that runs on CUDA and directs a smoothing filter using geometry buffers (G-buffers). It helps provide a smoother appearance in a pathtraced image with fewer samples per pixel and is based on the study "Edge-Avoiding A-Trous Wavelet Transform for rapid Global Illumination Filtering."

Features
=============
* A-Trous denoiser
* G Buffer visualization


Performance Analysis
============
**Additional Time for Each Frame**

The additional time varies from 1ms to ~10ms depending on the resolution. In a typical frame of 800x800, the average denoising time is 3ms.

**How denoising influences the number of iterations needed to get an "acceptably smooth" result**

A denoised image with 30 iteratiions could qualify the naive path traced image with 1000 iterations!

| 30 samples denoised image | 1000 samples naive image |
:-------:|:-------:
|![](img/denoiser/30sample_denoiser.png)|![](img/denoiser/1000sample.png) |

**How denoising at different resolutions impacts runtime**

From the table below, we could see that the time for denoising increases linearly as the resolution increases. However, when the filter size reaches 80, the time becomes almost constant. 

  | Denoising time vs Resolution |
|:--:|  
 |![image](img/denoiser/time.png)|
 
**How varying filter sizes affect performance**

The effect of filter size increases initially and goes down later. 


  | Denoising time vs Resolution |
|:--:|  
 |![image](img/denoiser/time_fs.png)|

**how visual results vary with filter size -- does the visual quality scale uniformly with filter size?**

We can see from the table below that while the filter size is below 30, the denoise effect is not significant enough. When the filter size goes large, the background is blurred. Therefore, the ideal range of an efficient filter size should be between 30 to 60.

| Filter Size | Image |
:-------:|:-------:
|5|![](img/denoiser/fs5.png) |
|10|![](img/denoiser/fs10.png) |
|20|![](img/denoiser/fs20.png) |
|30|![](img/denoiser/fs30.png) |
|40|![](img/denoiser/fs50.png) |
|50|![](img/denoiser/fs60.png) |
|60|![](img/denoiser/fs70.png) |
|70|![](img/denoiser/fs80.png) |
|80|![](img/denoiser/fs90.png) |
|90|![](img/denoiser/fs100.png) |
|100|![](img/denoiser/fs110.png) |

**how effective/ineffective is this method with different material types**

We can see that the denoiser works fine on alll types of materials except that the refractiion could be blurred a little bit. 

| Materials | Orignial Image| Denoised Image |
:-------:|:-------:|:-------:
|Diffuse|![](img/denoiser/diffuse_naive.png) |![](img/denoiser/diffuse_denoised.png) |
|Specular|![](img/denoiser/specular_naive.png) |![](img/denoiser/fs50.png) |
|Refractive|![](img/denoiser/refract_naive.png) |![](img/denoiser/refract_denoised.png) |

**How do results compare across different scenes - for example, between cornell.txt and cornell_ceiling_light.txt. Does one scene produce better denoised results? Why or why not?**

In my case, the denoiser works better in brighter scene. The reason is, with a light that has a larger areas, more paths fall onto the light source within a limited number of iterations and the scene converges faster. As a result, there are less noises inherently in the scene and leave less work for the denoiser whose performance could significantly be influenced by the initial conditions.  


**Other scenes**

We can see from the table below that the denoiser fails to work to render a bunny with ~50 samples. The reason could be that too many triangles with different information are concentreted in a small range. The problem can be mitigated when we switch to a small size filter.


 
| Bunny with 50 samples | Denosied Bunny with Filter Size 60 | Denosied Bunny with Filter Size 25 |
:-------:|:-------:|:-------:
|![](img/denoiser/bunny_naive.png)|![](img/denoiser/bunny_denoised.png) |![](img/denoiser/bunny_denoised25.png) |


**G Buffers**


| World Normal | Position |
:-------:|:-------:
|![](img/denoiser/normal.png)|![](img/denoiser/pos.png) |


Bloopers
===============
  | *G-Buffer Position Fail* |
|:--:|  
 |![image](img/bloopers/gbuffer_pos_bug2.png)|
 

  | *G-Buffer Position Fail2* |
  |:--:|
  |![image](img/bloopers/gbuffer_pos_bug3.png)|
   
Reference
===============
* [Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)

* [imgui]( https://github.com/ocornut/imgui)