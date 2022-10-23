CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Eyad Almoamen
  * [LinkedIn](https://www.linkedin.com/in/eyadalmoamen/), [personal website](https://eyadnabeel.com)
* Tested on: Windows 11, i7-10750H CPU @ 2.60GHz 2.59 GHz 16GB, RTX 2070 Super Max-Q Design 8GB (Personal Computer)

Introduction
================
I've built a GPU accelerated monte carlo path tracer using CUDA and C++. The parallelization is happening on a ray-by-ray basis, with the terminated rays being eliminated via stream compaction and sorted by material type in order to avoid warp divergence. The path tracer takes in a scene description .txt file and outputs a rendered image. 

Features implemented include:

* [Specular Reflective Material](#specular-reflective-material)
* [Refractive Material](#refractive-material)
* [Thin Lens Model DOF](#thin-lens-model-dof)
* [Motion Blur](#motion-blur)
* [Stochastic Antialiasing](#stochastic-antialiasing)
* [Direct Lighting](#direct-lighting)
* [Denoising](#denoising)

## Specular Reflective Material
================
The specular reflective material either reflects light perfectly (incident angle == exitent angle), or diffusely, the rate of each is manually set and the two percentages sum up to 100% (for example, if the material was 63% specular, it'd have to be 37% diffuse):

<img align="center" src="img/cornell.2022-10-11_03-01-03z.11379samp.png" width=50% height=50%>

#Refractive Material
================
The specular refractive material either reflects light or transmits it according to [Snell's Law](https://en.wikipedia.org/wiki/Snell%27s_law), the rate of each is based on the material type and index of refration. This is usually calculated by the [Fresnel Equations](https://en.wikipedia.org/wiki/Fresnel_equations), however, here I use the [Schlick approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation) to calculate the rates as it's more computationally efficient with a very low error rate:

<img align="center" src="img/cornell.2022-10-11_02-20-06z.5201samp.png" width=50% height=50%>

<img align="center" src="img/cornell.2022-10-11_00-50-38z.5598samp.png" width=50% height=50%>

## Thin Lens Model DOF
================
I utilized the [Thin Lens Model](https://pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#TheThinLensModelandDepthofField) in order to replace the pinhole camera we have with a more realistic virtual lens which allows me to introduce depth of field effects and bokeh:

| Focal Distance | 0 | 3 | 8.5 | 20.5 |
| :------- | :-------: | :-------: | :-------: | :-------: |
| Iterations | 7759 | 5082 | 5142 | 5009 |
| Scene | <img src="img/cornell.2022-10-11_02-43-13z.7759samp.png"> | <img src="img/cornell.2022-10-11_01-23-17z.5082samp.png"> | <img src="img/cornell.2022-10-10_23-09-12z.5142samp.png"> |  <img src="img/cornell.2022-10-11_01-07-49z.5009samp.png"> |

## Motion Blur
================
I added a velocity component to the geometry struct and that allows me to render the image in such a way that it seems the object is moving in the direction of the velocity:

## Stochastic Antialiasing
================
I added support for stochastic antialiasing by jittering the ray produced from the camera randomly within the range of a pixel length:

| Antialiasing | Without | With |
| :------- | :-------: | :-------: |
| Scene | <img src="img/cornell.2022-10-11_03-38-02z.1000samp.png"> | <img src="img/cornell.2022-10-11_03-40-14z.1000samp.png"> |
| Scene | <img src="img/cornell.2022-10-11_03-54-58z.1000samp.png"> | <img src="img/cornell.2022-10-11_03-53-19z.1000samp.png"> |

## Direct Lighting
================
To optimize the result and speed up the convergence of the image, I had the pathtracer trace its last ray to a light source in the scene, guaranteeing that we get light contribution. To demonstrate, I've rendered the same scene up to 1000 iterations with and without direct lighting:

| Direct Lighting | Without | With |
| :------- | :-------: | :-------: |
| Scene | <img src="img/"> | <img src="img/"> |

## Denoising
================
In order to be able to get an acceptable render faster, I've implemented the [Edge Avoiding À-Trous Wavelet Transform](https://jo.dreggn.org/home/2010_atrous.pdf) denoising function. The basic idea is to apply blur to the image while preserving edges between different objects and materials to create the impression of a converged image. Features such as geometry position, surface normal, and material color are used to detect edges between objects and from there apply a blur kernel to the image with varying filter sizes:

| Render | Distance | Position | Normal | Material Color |
| :------- | :-------: | :-------: | :------- | :-------: |
| <img src="img/cornelldenoise1000.png"> | <img src="img/distbuffer_cornell.png"> | <img src="img/posbuffer_cornell.png"> | <img src="img/norbuffer_cornell.png"> | <img src="img/colbuffer_cornell.png"> |
| <img src="img/3matdenoise1000.png"> | <img src="img/distbuffer_3mat.png"> | <img src="img/posbuffer_3mat.png"> | <img src="img/norbuffer_3mat.png"> | <img src="img/colbuffer_3mat.png"> | 
| <img src="img/thinlensdenoise1000.png"> | <img src="img/distbuffer_thinlens.png"> | <img src="img/posbuffer_thinlens.png"> | <img src="img/norbuffer_thinlens.png"> | <img src="img/colbuffer_thinlens.png"> | 

In general I've found that with filter size 2 or 3, I can get feasible results within 1000 iterations where it usually takes 5000:

| Scene | <img src="img/cornell1000.png"> | <img src="img/cornell5000.png"> | <img src="img/cornelldenoise1000.png"> |
| :------- | :-------: | :-------: | :------- |
| Denoised | No | No | Yes |
| Iterations | 1000 | 5000 | 1000 |

| Scene | <img src="img/3mat1000.png"> | <img src="img/3mat5000.png"> | <img src="img/3matldenoise1000.png"> |
| :------- | :-------: | :-------: | :------- |
| Denoised | No | No | Yes |
| Iterations | 1000 | 5000 | 1000 |

| Scene | <img src="img/thinlens1000.png"> | <img src="img/thinlens5000.png"> | <img src="img/thinlensdenoise1000.png"> |
| :------- | :-------: | :-------: | :------- |
| Denoised | No | No | Yes |
| Iterations | 1000 | 5000 | 1000 |

Here I've rendered the same scene with different filter sizes to illustrate the effect. All renders ran for 1000 iterations:

| Filter Size | 1 | 2 | 3 | 4 | 8 |
| :------- | :-------: | :-------: | :------- | :-------: | :-------: |
| Scene | <img src="img/filtersize1.png"> | <img src="img/filtersize2.png"> | <img src="img/filtersize3.png"> | <img src="img/filtersize4.png"> | <img src="img/filtersize8.png"> |

As we can see here, one of the limitations of this approach is loss of data when it comes to specular surfaces or refractive surfaces, although this can be potentially counteracted by making more bounces for specular materials until we reach a diffuse surface. Of course if the surface is imperfect specular there's no need to do that as the result won't vary that much.

Performance Testing
================
I ran a few tests to see the effect of some of the optimizations I've performed on this path tracer:

The effect of caching is very much evident and it increases as the size of the image increases:

<img align="center" src="img/cachingchart.png" width=50% height=50%>

This is because we're precomputing a potentially very large computation, sparing ourselves the trouble for upcoming iterations

The effect of material sorting doesn't seem to be too encouraging; initially I tried testing it on a scene with one material, it wasn't an improvement (since we'd be sorting to avoid nonexistent warp divergence). However I switched to a scene with diffuse, reflective, and refractive material to no avail:

<img align="center" src="img/materialsortchart.png" width=50% height=50%>

Denoising the image predictably increases running time (assuming all else is constant) which makes sense since for every iteration, it has to apply the kernel and weights to the image. The results are as follows:

<img align="center" src="img/denoiserperformance.png" width=50% height=50%>

The results are more or less what you'd expect, approximately a linear factor slower with denoising. After that, I tested the effect of filter size on running time. My expectation was to see running time increasing linearly with filter size since you have to do that same amount of computation once more for every time you increment filter size. While I was testing filter size, I also tested the effect of using `#pragma unroll` on the `atrousDenoise()` function. The results were as follows:

<img align="center" src="img/filtersize.png" width=50% height=50%>

Initially, as I was increasing filter size, the amount of time it took was relatively flat. While that could've been linear with a small factor, it could just be measurement error. So I started using logarithmic intervals, and for a few iterations, it remained flat. When I hit filter size 32 I started to see a solid increase in running time, however it seemed it could be linear (on logarithmic intervals), however when I reached filter size 128, the exponential increase started to become clearer. So it seems the relationship between filter size and running time is linear with a small constant factor. 

As far as unrolling goes, it did introduce a consistent imrovement in running time, however it wasn't very substantial.
