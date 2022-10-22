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

## Core Features

## Performance Analysis

## Bloopers! :)

