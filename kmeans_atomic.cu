#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_POINTS 100000000
#define MAX_MEANS 1000
#define MAX_ITER 100

typedef struct {
  double *x, *y;
  int *membership;
} points;

typedef struct {
  double *x, *y;
  int *size;
  double *x_sum, *y_sum;
} centroids;

__managed__ int assignment_changed;

// reads n data points from input file
__host__ void read_data(int n, char *file_name, points P) {
  size_t i = 0;
  double x, y;
  FILE *file = fopen(file_name, "r");
  assert(file != NULL);

  while (!feof(file) && i < n) {
    if (fscanf(file, "%lf %lf", &x, &y) != 2)
      break;
    P.x[i] = x;
    P.y[i++] = y;

    if (i % (n / 100) == 0) {
      printf("\rReading input: %d%%", 100 * i / n);
      fflush(stdout);
    }
  }

  printf("Read %d points\n", i);
}

// selects k centers at random from n points
__host__ void init_centers(int n, int k, points P, centroids C) {
  for (int i = 0; i < k; ++i) {
    // not actually uniform random sampling, but good enough for testing
    int rand_idx = rand() % n;
    C.x[i] = P.x[rand_idx];
    C.y[i] = P.y[rand_idx];
  }
}

// computes ||p-c||^2 for a point p and center c
__device__ inline double norm_2D_sqr(double x1, double y1, double x2,
                                     double y2) {
  // sqrt is monotonic, so we may omit it in the distance calculation
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

__global__ void
assign_clusters(int n, int k, const double *__restrict__ Px,
                const double *__restrict__ Py, int *__restrict__ Pmembership,
                double *__restrict__ Cx, double *__restrict__ Cy,
                int *__restrict__ Csize, double *__restrict__ Cx_sum,
                double *__restrict__ Cy_sum) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < n) {
    double min_dist = INFINITY;
    int membership = -1;

    for (int i = 0; i < k; ++i) {
      double current_dist = norm_2D_sqr(Px[index], Py[index], Cx[i], Cy[i]);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        membership = i;
      }
    }

    if (membership != Pmembership[index])
      assignment_changed = 1;
    Pmembership[index] = membership;

    atomicAdd(&Cx_sum[membership], Px[index]);
    atomicAdd(&Cy_sum[membership], Py[index]);
    atomicAdd(&Csize[membership], 1);
  }
}

__global__ void update_clusters(int n, int k, double *__restrict__ Cx,
                                double *__restrict__ Cy,
                                double *__restrict__ Cx_sum,
                                double *__restrict__ Cy_sum,
                                int *__restrict__ Csize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < k && Csize[index]) {
    Cx[index] = Cx_sum[index] / Csize[index];
    Cy[index] = Cy_sum[index] / Csize[index];
  }
}

__global__ void zero_centroid_vals(int k, double *__restrict__ Cx_sum,
                                   double *__restrict__ Cy_sum,
                                   int *__restrict__ Csize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < k) {
    Cx_sum[index] = 0;
    Cy_sum[index] = 0;
    Csize[index] = 0;
  }
}

__host__ void print_results(int k, int n, int h, double t, points P,
                            centroids C) {
  printf("performed %d iterations in %.2f s, perf: %.2f billion\n", h, t,
         (double)k * n * h / t * 1e-9);

  for (int i = 0; i < k; ++i) {
    printf("cluster %d centered at (%f, %f) has size %d\n", i, C.x[i], C.y[i],
           C.size[i]);
  }
}

int main(int argc, char **argv) {
  int k, n, h;
  char *file_name;
  points P;
  centroids C;

  assert(argc >= 4);
  n = atoi(argv[1]);
  k = atoi(argv[2]);
  file_name = argv[3];
  assert(n <= MAX_POINTS && k <= MAX_MEANS);

  cudaMallocManaged(&P.x, sizeof(double) * n);
  cudaMallocManaged(&P.y, sizeof(double) * n);
  cudaMallocManaged(&P.membership, sizeof(int) * n);
  cudaMallocManaged(&C.x, sizeof(double) * k);
  cudaMallocManaged(&C.y, sizeof(double) * k);
  cudaMallocManaged(&C.size, sizeof(int) * k);
  cudaMallocManaged(&C.x_sum, sizeof(double) * k);
  cudaMallocManaged(&C.y_sum, sizeof(double) * k);

  read_data(n, file_name, P);
  init_centers(n, k, P, C);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  clock_t start = clock();

  for (h = 0; h < MAX_ITER; ++h) {
    assignment_changed = 0;
    zero_centroid_vals<<<1, k>>>(k, C.x_sum, C.y_sum, C.size);
    cudaDeviceSynchronize();

    assign_clusters<<<numBlocks, blockSize>>>(n, k, P.x, P.y, P.membership, C.x,
                                              C.y, C.size, C.x_sum, C.y_sum);
    cudaDeviceSynchronize();

    if (!assignment_changed)
      break;

    update_clusters<<<1, k>>>(n, k, C.x, C.y, C.x_sum, C.y_sum, C.size);
    cudaDeviceSynchronize();
  }

  clock_t end = clock();
  double t = (double)(end - start) / CLOCKS_PER_SEC;

  print_results(k, n, h, t, P, C);
}
