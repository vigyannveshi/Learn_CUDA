import numpy as np

n=5000
mat1=np.arange(0,n*n).reshape((n,n))
print(mat1@mat1)