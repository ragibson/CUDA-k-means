import matplotlib.pyplot as plt

K = 1000
M = 1000 * K

xs = [10 * K, 50 * K, 100 * K, 1 * M, 5 * M, 10 * M, 25 * M, 50 * M, 100 * M]
y1 = [0.6, 2.72, 4.90, 15.89, 25.99, 31.45, 31.84, 32.89, 34.39]
y2 = [1.31, 6.44, 12.09, 40.34, 58.04, 59.14, 60.37, 61.34, 61.69]
y3 = [2.03, 9.47, 16.96, 47.49, 63.25, 63.08, 63.64, 63.85, 64.01]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, y1, "o-", markersize=5, label=r"k=10", color="C0")
plt.plot(xs, y2, "o-", markersize=5, label=r"k=100", color="C1")
plt.plot(xs, y3, "o-", markersize=5, label=r"k=1000", color="C2")
plt.xscale('log', nonposx='clip')
plt.title(r"Parallel $k$-means simulation performance, $h_{max} = 30$, varying $n$")
plt.xlabel("Number of points, $n$")
plt.ylabel("Billions of interactions per second (average)")
plt.ylim([0, 70])
plt.legend()
plt.tight_layout()
plt.savefig("perf_plot1_log.png", dpi=200)

plt.close()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xs, y1, "o-", markersize=5, label=r"k=10", color="C0")
plt.plot(xs, y2, "o-", markersize=5, label=r"k=100", color="C1")
plt.plot(xs, y3, "o-", markersize=5, label=r"k=1000", color="C2")
plt.title(r"Parallel $k$-means simulation performance, $h_{max} = 30$, varying $n$")
plt.xlabel("Number of points, $n$")
plt.ylabel("Billions of interactions per second (average)")
plt.ylim([0, 70])
plt.legend()
plt.tight_layout()
plt.savefig("perf_plot1_lin.png", dpi=200)
