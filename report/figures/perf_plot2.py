import matplotlib.pyplot as plt

K = 1000
M = 1000 * K

xs = [2, 4, 5, 10, 25, 50, 100, 250, 500, 1000]
y1 = [0.92, 0.76, 2.18, 15.76, 25.53, 33.88, 40.47, 45.33, 48.19, 47.62]
y2 = [0.74, 2.88, 3.54, 23.94, 45.65, 53.74, 60.04, 63.13, 64.21, 63.15]
y3 = [1.44, 3.67, 4.54, 33.75, 49.06, 57.04, 61.65, 64.25, 65.67, 64.00]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, y1, "o-", markersize=5, label=r"n=1M", color="C0")
plt.plot(xs, y2, "o-", markersize=5, label=r"n=10M", color="C1")
plt.plot(xs, y3, "o-", markersize=5, label=r"n=100M", color="C2")
plt.xscale('log', nonposx='clip')
plt.title(r"Parallel $k$-means simulation performance, $h_{max} = 30$, varying $k$")
plt.xlabel("Number of means, $k$")
plt.ylabel("Billions of interactions per second (average and range)")
plt.ylim([0, 70])
plt.legend()
plt.tight_layout()
plt.savefig("perf_plot2_log.png", dpi=200)

plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, y1, "o-", markersize=5, label=r"n=1M", color="C0")
plt.plot(xs, y2, "o-", markersize=5, label=r"n=10M", color="C1")
plt.plot(xs, y3, "o-", markersize=5, label=r"n=100M", color="C2")
plt.title(r"Parallel $k$-means simulation performance, $h_{max} = 30$, varying $k$")
plt.xlabel("Number of means, $k$")
plt.ylabel("Billions of interactions per second (average)")
plt.ylim([0, 70])
plt.legend()
plt.tight_layout()
plt.savefig("perf_plot2_lin.png", dpi=200)
