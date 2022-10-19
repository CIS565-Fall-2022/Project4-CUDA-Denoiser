CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Alex Fu
  
  - [LinkedIn](https://www.linkedin.com/in/alex-fu-b47b67238/)
  - [Twitter](https://twitter.com/AlexFu8304)
  - [Personal Website](https://thecger.com/)
  
  Tested on: Windows 10, i7-10750H @ 2.60GHz, 16GB, GTX 3060 6GB
* [repo link](https://github.com/IwakuraRein/Nagi)

## Features

A real-time path tracing denoiser. Reference: [*Spatiotemporal variance-guided filtering: real-time reconstruction for path-traced global illumination*](https://dl.acm.org/doi/10.1145/3105762.3105770).

<video src="https://user-images.githubusercontent.com/28486541/196747599-32b3307a-4af8-43af-bf47-4a27321f0234.mp4"></video>

## G-Buffer optimization

In order to represent the geometries of reflection, I blend the geometries according to material types:

Diffuse material:

* Albedo buffer: store first bounce albedo

* Normal buffer: store first bounce normal

* Depth buffer: store first bounce depth

Glossy Material:

- Albedo buffer: blend the first and second bounce albedo

- Normal buffer: store first bounce normal

- Depth buffer: store first bounce depth

Specular Material:

* Albedo buffer: blend the albedo until hit non-specular material

* Normal buffer: store the first non-specular material

* Dpeth buffer: accumulate depth until hit non-specular material

