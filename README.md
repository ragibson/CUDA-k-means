# CUDA-k-means

This repository contains three main implementation of Lloyd's k-means
algorithm:

* [kmeans_cpu.c](kmeans_cpu.c): A k-means implementation for use on a single
CPU core.
* [kmeans_gpu.cu](kmeans_gpu.cu): A k-means implementation for use on CUDA
GPUs.
* [kmeans_atomic.cu](kmeans_atomic.cu): A k-means implementation for use on
CUDA GPUs, using atomic operations (for comparison purposes).

In terms of "number of cluster updates per second", these programs
peak at

* [kmeans_cpu.c](kmeans_cpu.c): ~950 million computations per second
* [kmeans_gpu.cu](kmeans_gpu.cu): ~65 billion computations per second
* [kmeans_atomic.cu](kmeans_atomic.cu): ~575 billion computations per second
(though this is only feasible for a large number of clusters)

on a system with an E5-2650v3 CPU and TITAN V GPU.

A (significantly) more detailed performance analysis can be read
[here](report/report.pdf).
