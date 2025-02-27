import numpy as np

# Task 2
N = 10
A = np.diag(np.arange(1,N),-1)
A = np.flip(A, axis=1)

print(A)