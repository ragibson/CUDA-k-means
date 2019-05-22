import numpy as np
import sys

# 10 20 5 2 2000000 0.1 generates 100M points
num_branches1 = int(sys.argv[1])
dist_branches1 = float(sys.argv[2])
num_branches2 = int(sys.argv[3])
dist_branches2 = float(sys.argv[4])
cluster_size = int(sys.argv[5])
cluster_scale = float(sys.argv[6])

for i in range(num_branches1):
    for j in range(num_branches2):
        val = dist_branches1 * 1j ** (i * 4 / num_branches1) + \
              dist_branches2 * 1j ** (j * 4 / num_branches2)
        for _ in range(cluster_size):
            x = val.real + np.random.normal(scale=cluster_scale)
            y = val.imag + np.random.normal(scale=cluster_scale)
            print(x, y)
