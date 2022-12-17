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
I compared the result image with denoiser on and off. 

| |  10 iterations | Orthogonal view of New York |
|Denoiser on|----------------- | ----------------- | 
|Denoiser off |![](images/result/newYork/newYorkAzimuth.png) | ![](images/result/newYork/newYorkImage.png)  |  
* Impact of Different Resolutions on Runtime  

* Impact of filter sizes on Runtime  



* Impact of filter sizes on Image  

* Effectiveness of this method on different materials  

* Results across Different Scenes  
