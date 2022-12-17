CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* XiaoyuDu
* Tested on: Windows 10, i9-11900KF @ 3.50GHz, RTX 3080 (Personal PC)

  
### Description  
This project built a denoiser based on an GPU-based path tracer. The algorithm of the denoiser follows the mthods proposed in the paper "Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering" by Holger Dammertz, Daniel Sewtz, Johannes Hanika, Hendrik P.A. Lensch. 
  
  
### Feature  
* I implemented all the features for part 1.  


### Analysis  
* Runtime for Denoiser  
I tested the denoiser with 10 iterations of the cornell box scene. Averagely speaking, after adding the denoiser, the runtime for each iteration increased for about 1.11242ms.  
* Influence on the number of iterations  
I tested the influence of denoiser with filter size of 20, color weight of 12, normal weight of 0.02, and position weight of 0.34. The results are shown below. As you can see, originally, we need around 500 iterations to get an "acceptably smooth" result. With this denoiser, only around 60 iterations are needed to get an "acceptably smooth" result.  
|  Denoiser on | Denoiser off |
|----------------- | ----------------- | 
|![](images/cornell_ceiling/1_d.png) | ![](images/cornell_ceiling/29.png)  |  
|![](images/cornell_ceiling/6_d.png) | ![](images/cornell_ceiling/99.png)  |  
|![](images/cornell_ceiling/60_d.png) | ![](images/cornell_ceiling/500.png)  |
  
* Impact of Different Resolutions on Runtime  

* Impact of filter sizes on Runtime  



* Impact of filter sizes on Image  

* Effectiveness of this method on different materials  

* Results across Different Scenes  
