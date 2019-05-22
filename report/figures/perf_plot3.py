import matplotlib.pyplot as plt

K = 1000
M = 1000 * K

xs = [2, 4, 5, 10, 15, 20, 25, 30]
y1 = [23.17, 33.61, 37.36, 49.73, 54.14, 54.38, 57.37, 56.31]
y2 = [24.37, 39.42, 42.13, 52.79, 55.69, 57.93, 59.21, 60.61]
y3 = [19.02, 25.93, 28.94, 34.12, 37.09, 38.35, 39.49, 39.75]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, y3, "o-", markersize=5, label=r"n=1M, k=100", color="C0")
plt.plot(xs, y2, "o-", markersize=5, label=r"n=10M, k=100", color="C1")
plt.plot(xs, y1, "o-", markersize=5, label=r"n=50M, k=100", color="C2")
plt.xscale('log', nonposx='clip')
plt.title(r"Parallel $k$-means simulation performance, varying $h$")
plt.xlabel("Number of maximum iterations, $h$")
plt.ylabel("Billions of interactions per second (average and range)")
plt.ylim([0, 70])
plt.legend()
plt.tight_layout()
plt.savefig("perf_plot3_log.png", dpi=200)

plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, y3, "o-", markersize=5, label=r"n=1M, k=100", color="C0")
plt.plot(xs, y2, "o-", markersize=5, label=r"n=10M, k=100", color="C1")
plt.plot(xs, y1, "o-", markersize=5, label=r"n=50M, k=100", color="C2")
plt.title(r"Parallel $k$-means simulation performance, varying $h$")
plt.xlabel("Number of maximum iterations, $h$")
plt.ylabel("Billions of interactions per second (average)")
plt.ylim([0, 70])
plt.legend()
plt.tight_layout()
plt.savefig("perf_plot3_lin.png", dpi=200)
