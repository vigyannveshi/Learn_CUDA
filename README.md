## **Learn CUDA**

Notes and programs written as I work through
[Mastering GPU Parallel Programming with CUDA: ( HW & SW )](https://www.udemy.com/course/mastering-gpu-parallel-programming-with-cuda)
by Hamdy Sultan. Content reflects my own understanding and grows with the course.

### **Environment**
- CUDA Toolkit 12.1 (via conda)
- Python 3.10
- NVIDIA-RTX 3090

### **Setup**
```bash
conda create --prefix /path/to/learn_cuda python=3.10 -y
conda activate /path/to/learn_cuda
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y
```

### **Structure**
- `src/` — CUDA source files (.cu)
- `NOTES.md` — learning notes

### **Contents**
| File | Topic |
|------|-------|
| src/001_hello_world.cu | Thread and block IDs |
| src/002_hello_world.cu | Warp IDs |