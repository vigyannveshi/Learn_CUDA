## **Compute Unified Device Architecture (CUDA)**

### **Installations:**

**1. Creating conda environment**
```
    conda create --prefix conda_env_path/learn_cuda python=3.10 -y
```

**2. Installing CUDA toolkit**
```
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y
```

### **Introduction**
* Parallel computing platform and application programming inferface (API).
* GPGPU (General-Purpose computing on Graphics Processing Units).
* Based on C-programming language.
* CUDA compiler is called NVCC (Nvidia-CUDA Compiler).
* Host and Device
  * Host: CPU + DRAM.
  * Device: Nvidia-GPU + GDRAM .

* Hardware has 4 levels <--> Software
  1. GPU itself   < -- > Group of Blocks/Thread Blocks.
  2. Streaming Multiprocessors (SMs) < -- > each block/ thread block.
  3. Partitions (eg: Volta and Ampere architecture has 4 such partitions per SM) < -- > warps.
  4. Individual Computational units within each partition (granular level), eg: FP32 core <--> thread.

* GigaThread Engine allocates tasks/jobs/thread blocks to SMs
* Warp schedular is present in each partition which schedules/assigns warps to partitions within an SM during each cycle.
* Each block is sub-divided into warps. 
* A warp typically consists of 32 threads.

**CUDA programming**

* A kernel in CUDA <--> function in C/C++ to be executed on GPU.
* `__global__` is used as return type for kernels.
* Compiling the program `nvcc input_path/filename.cu -o output_path/filename_out.cu` 
* Executing the application `./output_path/filename_out.cu`