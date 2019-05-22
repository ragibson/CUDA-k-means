import matplotlib.pyplot as plt
import sys

xs = []
ys = []

for line in sys.stdin:
    if line[0] == '=':  # new cluster
        plt.scatter(xs, ys, s=10)
        plt.plot([float(line.split(" ")[4])],
                 [float(line.split(" ")[5])],
                 color='black', marker='x')
        xs = []
        ys = []
    else:
        xs.append(float(line.split(" ")[0]))
        ys.append(float(line.split(" ")[1]))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.scatter(xs, ys, s=10)
plt.title("Example output when $k=30$")
plt.tight_layout()
plt.savefig("Ex_k=30.png", dpi=200)
