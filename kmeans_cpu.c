#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_POINTS 10000000
#define MAX_MEANS 10000
#define MAX_ITER 100

typedef struct {
  double x[MAX_POINTS];
  double y[MAX_POINTS];
  int membership[MAX_POINTS];
} points;

typedef struct {
  double x[MAX_MEANS];
  double y[MAX_MEANS];
  int cluster_size[MAX_MEANS];
} centers;

int n;
int k;
points P;
centers C;

void generate_data() {
  srand(time(NULL));

  for (int i = 0; i < n; ++i) {
    double rand_x = (double)rand() / RAND_MAX;
    double rand_y = (double)rand() / RAND_MAX;
    P.x[i] = rand_x;
    P.y[i] = rand_y;
    P.membership[i] = 0;
  }
}

inline double point_to_center_dist(int point_idx, int center_idx) {
  double xdist = P.x[point_idx] - C.x[center_idx];
  double ydist = P.y[point_idx] - C.y[center_idx];
  // sqrt is monotonic, so we may omit it in the distance calculation
  return xdist * xdist + ydist * ydist;
}

int assign_clusters() {
  int assignment_changed = 0;

  for (int i = 0; i < k; ++i)
    C.cluster_size[i] = 0;

  for (int i = 0; i < n; ++i) {
    int old_assignment = P.membership[i];
    double min_dist = point_to_center_dist(i, old_assignment);

    for (int j = 0; j < k; ++j) {
      double current_dist = point_to_center_dist(i, j);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        P.membership[i] = j;
      }
    }

    C.cluster_size[P.membership[i]]++;
    assignment_changed |= (P.membership[i] != old_assignment);
  }

  return assignment_changed;
}

void update_clusters() {
  for (int i = 0; i < k; ++i) {
    if (C.cluster_size[i]) {
      C.x[i] = 0;
      C.y[i] = 0;
    }
  }

  for (int i = 0; i < n; ++i) {
    C.x[P.membership[i]] += P.x[i];
    C.y[P.membership[i]] += P.y[i];
  }

  for (int i = 0; i < k; ++i) {
    if (C.cluster_size[i]) {
      C.x[i] /= C.cluster_size[i];
      C.y[i] /= C.cluster_size[i];
    }
  }
}

int main(int argc, char **argv) {
  int h;

  if (argc < 3) {
    fprintf(stderr, "Too few arguments.\n");
    exit(1);
  }

  n = atoi(argv[1]);
  k = atoi(argv[2]);

  if (n > MAX_POINTS || k > MAX_MEANS) {
    if (n > MAX_POINTS)
      fprintf(stderr, "n = %d is greater than MAX_POINTS = %d.\n", n,
              MAX_POINTS);
    if (k > MAX_MEANS)
      fprintf(stderr, "k = %d is greater than MAX_MEANS = %d.\n", k, MAX_MEANS);
    exit(1);
  }

  generate_data();

  for (int i = 0; i < k; ++i) {
    // not actually uniform random sampling, but good enough for testing
    int rand_idx = rand() % n;
    C.x[i] = P.x[rand_idx];
    C.y[i] = P.y[rand_idx];
    C.cluster_size[i] = 0;
  }

  clock_t start = clock();

  for (h = 0; assign_clusters() && h < MAX_ITER; ++h)
    update_clusters();

  clock_t end = clock();
  double t = (double)(end - start) / CLOCKS_PER_SEC;

  printf("performed %d iterations in %.2f s, perf: %.2f million\n", h, t,
         (double)k * n * h / t * 1e-6);

  /*for (int i = 0; i < k; ++i) {
    printf("cluster %d centered at (%f, %f) has size %d\n", i, centers[i].x,
           centers[i].y, centers[i].cluster_size);
  }

  printf("membership: ");
  for (int i = 0; i < n; ++i) {
    printf("%d ", points[i].membership);
  }
  printf("\n");*/
}
