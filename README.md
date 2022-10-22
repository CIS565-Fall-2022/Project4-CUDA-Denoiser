CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Nick Moon
  * [LinkedIn](https://www.linkedin.com/in/nick-moon1/), [personal website](https://nicholasmoon.github.io/)
* Tested on: Windows 10, AMD Ryzen 9 5900HS @ 3.0GHz 32GB, NVIDIA RTX 3060 Laptop 6GB (Personal Laptop)


**This project is an implementation of the Edge-Avoiding À-Trous Wavelet Transform for Fast Global
Illumination Filtering.
This denoising algorithm uses a style of gaussian blurring to smooth noisy parts of 
the render, while smartly detecting edges with G-Buffer values stored during path-tracing.
This allows for segmented denoising that preserves object boundaries.**

## RESULTS


| Denoised 1 SPP   | Denoised 100 SPP | Denoised 1000 SPP |
| ----------- | ----------- |  ----------- |
| ![](img/results/render_denoised_1.PNG)      |   ![](img/results/render_denoised_100.PNG)     | ![](img/results/render_denoised_1000.PNG) |

| Original 1 SPP   | Original 100 SPP | Original 1000 SPP |
| ----------- | ----------- |  ----------- |
| ![](img/results/render_1.PNG)      |   ![](img/results/render_100.PNG)     | ![](img/results/render_1000.PNG) |

Adding denoising to these renders only incurred an additional constant 27ms of runtime, no
matter how many path tracing iterations were used!


## IMPLEMENTATION

### Gaussian Blur and Filtering

As a small introduction, the core of the denoising algorithm is based on filters/kernels. These
are a collection of values that describe weighting around a center pixel ```p```. 
So, for example, if you have a 5x5
kernel ```k``` and are at pixel ```p``` then the middle element ```k[2][2]``` will be multiplied
by the value at pixel p, and the result accumulated for each of the 25 pixels around ```p```.

Below is an example of a kernel generated with the gaussian function:

![](img/figures/gaussiankernel.PNG)

Applying the kernel to every pixel in an image will result in blur like the image below:

| Original   | Blurred |
| ----------- | ----------- |
| ![](img/results/iteration_1.PNG)      |   ![](img/figures/gaussian_blur.png)     |



### À-Trous Wavelet Transform

The À-Trous Wavelet Transform described in the paper is a filter similar to the gaussian kernel, 
but optimized. Instead of having a kernel size that grows quadratically with the number of
pixels desired to be sampled, the À-Trous Wavelet Transform instead reuses the same kernel,
for example a 5x5 like used in this project, but performs multiple iterations of denoising
using exponentially greater offsets between pixels sampled each time. This allows for a larger
neighborhood of pixels to be sampled without significantly increasing the amount of computation
required. An illustration of this is shown in the below figure:

![](img/figures/kerneloffsets.PNG)

Below is also a demonstration of the À-Trous Wavelet Transform applied to a noisy path-traced
cornell box render, without the edge detection described in the next section:

| Kernel Size 1 (1 iter)    | Kernel Size 4 (3 iter) | Kernel Size 16 (5 iter) | Kernel Size 64 (7 iter) |
| ----------- | ----------- |  ----------- |  ----------- |
| ![](img/results/no_edge_detection_filter1.PNG)      |   ![](img/results/no_edge_detection_filter3.PNG)     | ![](img/results/no_edge_detection_filter5.PNG) | ![](img/results/no_edge_detection_filter7.PNG) |

As can be seen, this looks very similar to the pure gaussian blur from the previous section.
It just blurs the entire screen, and it would be rare to describe it as "denoising".

Specifically, the offset between pixels for each iteration ```i``` of the kernel is ```2^i```.


### Edge Detection

#### G-Buffer

The edge detection process uses the positions and normals at the intersection of the 
camera rays associated with each pixel. So, in order to have this information available to us
to use in post process, we need to create a new geometry buffer (G-Buffer) to store relevant 
information at each pixel.

Below you can see a visualization of the data collected in this G-Buffer for a simple scene:

| Position Buffer      | Normal Buffer | Depth Buffer |
| ----------- | ----------- | ----------- |
| ![](img/results/pos_buffer.PNG)     | ![](img/results/nor_buffer.PNG)       | ![](img/results/depth_buffer.PNG) |

#### Edge Detecting with Weights

Edge detection is perfomed by using the source path-traced image (i.e. per-pixel color information),
per-pixel intersection world space positions, and per-pixel intersection world space normals.
At a certain pixel ```p```, the squared distance between ```p's``` position, normals, and 
color information and one of ```p's``` neighbors (what index into the filter the process is in)
is calculated. Then weighting terms for these three components (color, position, and normal)
are calculated using an exponential function. The three weights are multiplied together
to get a combined weight for this pixel comparison. 

## Visual Analysis


### Filter Size
Below is a visual comparison of different filter sizes with edge detection. The number of
sample-per-pixel is only 20, with a very high color weight.

| Max Kernel Size 1     | Max Kernel Size 2 | Max Kernel size 4 |
| ----------- | ----------- |  ----------- |
| ![](img/results/kernel_size_1_iter_1.PNG)      |   ![](img/results/kernel_size_2_iter_1.PNG)     | ![](img/results/kernel_size_3_iter_1.PNG) |

| Max Kernel Size 8     | Max Kernel Size 16 | Max Kernel size 32 |
| ----------- | ----------- |  ----------- |
| ![](img/results/kernel_size_4_iter_1.PNG)      |   ![](img/results/kernel_size_5_iter_1.PNG)     | ![](img/results/kernel_size_6_iter_1.PNG) |

| Max Kernel Size 64     | Max Kernel Size 128 | Max Kernel size 256 |
| ----------- | ----------- |  ----------- |
| ![](img/results/kernel_size_7_iter_1.PNG)      |   ![](img/results/kernel_size_8_iter_1.PNG)     | ![](img/results/kernel_size_9_iter_1.PNG) |

### Different Material Types

| 1 Iteration Not Denoised     | 1 Iteration Denoised | 5000 Iterations Not Denoised |
| ----------- | ----------- |  ----------- |
| ![](img/results/matcom_1iter.PNG)      |   ![](img/results/matcomp_denoised.PNG)     | ![](img/results/matcomp_5000iter.PNG) |

As can be seen from the comparison above, the denoising algorithm actually struggles a bit with specular
materials. While the edge detection handles the edges of the specular material quite well, the
reflection on the surface appears rough and more like a microfacet material. This also does not
go completely away until hundreds of iterations of path tracing are generated. Unlike the
specular surface, the diffuse surface already scatters light in all directions randomly, so
the smudging and blurring is MUCH less apparant. At one iteration and denoised the sphere
almost looks good enough to consider converged.

### Different Scenes

|  | 1 Iteration  | 100 Iterations |
| ----------- | ----------- | ----------- |
| Smaller Ceiling Light     |   ![](img/results/cornell_1iter_denoised.PNG)     | ![](img/results/cornell_100iter_denoised.PNG) |
| Larger Ceiling Light     |   ![](img/results/iteration_1_denoised.PNG)     | ![](img/results/iteration_100_denoised.PNG) |

As can be seen above, the denoising algorithm struggles a lot more with the smaller light scene
than the larger light scene. This is because the smaller light scene will naturally sample the
light less times than the one with a larger light, because it is much more likely to hit the
larger light while sending a ray in a random direction. This of course impacts the denoising,
because at lower path-tracing iterations, this will mean more pixels are black, and also
there will be much more variance between pixels. Both of these are bad for blurring, because
blurring requires there to be at least some minimum amount of useful information to use
without looking splotchy (almost like low iteration photon mapping).

## Performance Analysis

### Convergence
Below shows renders of different samples-per-pixel before and after denoising:

| 1 Iteration     | 5 Iterations  | 10 Iterations |
| ----------- | ----------- |  ----------- |
| Original     | Original  | Original |
| ![](img/results/iteration_1.PNG)      |   ![](img/results/iteration_5.PNG)     | ![](img/results/iteration_10.PNG) |
| Denoised     | Denoised  | Denoised |
| ![](img/results/iteration_1_denoised.PNG)      |   ![](img/results/iteration_5_denoised.PNG)     | ![](img/results/iteration_10_denoised.PNG) |

| 50 Iteration     | 100 Iterations  | 500 Iterations |
| ----------- | ----------- |  ----------- |
| Original     | Original  | Original |
| ![](img/results/iteration_50.PNG)      |   ![](img/results/iteration_100.PNG)     | ![](img/results/iteration_500.PNG) |
| Denoised     | Denoised  | Denoised |
| ![](img/results/iteration_50_denoised.PNG)      |   ![](img/results/iteration_100_denoised.PNG)     | ![](img/results/iteration_500_denoised.PNG) |

| 1000 Iteration     | 5000 Iterations  |
| ----------- | ----------- |
| Original     | Original  |
| ![](img/results/iteration_1000.PNG)      |   ![](img/results/iteration_5000.PNG)     |
| Denoised     | Denoised  |
| ![](img/results/iteration_1000_denoised.PNG)      |   ![](img/results/iteration_5000_denoised.PNG)     |


I would say that, for denoising, iteration 500 is about where I would say the results are "acceptably smooth".
And what I mean by that, is that by iteration 500 the image not only looks like the background colors
have smooth outed to a near converged look, but also that the specular sphere no longer looks smudged.
In comparison, I think that between 1000-5000 iterations is where the none denoised render
looks "acceptably smooth". Anything before that has the apparant path-tracing noise pattern.
Here are the diff images for 500 iterations with and without denoising in comparison to
the 5000 iteration result (with no denoising):

| | Original     | Denoised  |
| ----------- | ----------- | ----------- |
| Render | ![](img/results/iteration_500.PNG)      |   ![](img/results/iteration_500_denoised.PNG)     |
| Diff from 5000 iter | ![](img/results/diff_500.PNG)      |   ![](img/results/diff_500_denoised.PNG)     |

### Varying Filter Size

![](img/figures/denoise_runtime_vs_pt_iter.png)

As can be seen from the graph above, the amount of time taken for the denoising algorithm
with varying number of path tracing iterations is the same; this is because the denoising
algorithm is only influenced by the resolution of the image to denoise and the size of the 
convolution filter. In addition, each increase in filter size only results in a constant amount
of additional runtime, about equal to the size of filter size 1. This is because the number of
pixels we are sampling for each additional iteration of the denoising kernel is the same,
as a result of the A Trous Wavelet. This means that this algorithm scales very well regardless
of number of samples taken.

### Path Tracing vs Denoising

![](img/figures/pathtracing_v_denoising.png)

As can be seen from the figure above, and having shown already that the denoising algorithm
runtime is independent of the number of path tracing iterations and indeed constant at a
certain filter size, the percentage of time taken to do the denoising operation vs. the actual
path tracing decreases exponentially, and the dropoff is fast. This is because each path tracing
iteration takes about 7 seconds to run, about the same time as the denoising algorithm. So, each
additional iteration of path tracing cuts the percentage of time taken to do the denoising
be around ```1 / iter + 1```.

### Render Resolution

![](img/figures/resolution.png)

As can be seen from the figure above, the runtime of the denoising algorithm increases about
quadratically with increased resolution (where width and height are the same). This is what we
expect. Although the kernel size is constant for all of these data points, the number of pixels
the GPU needs to run the code on increases quadratically as well.

## References

## Performance Analysis

Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering:

Paper: https://jo.dreggn.org/home/2010_atrous.pdf

Presentation: https://www.highperformancegraphics.org/previous/www_2010/media/RayTracing_I/HPG2010_RayTracing_I_Dammertz.pdf

