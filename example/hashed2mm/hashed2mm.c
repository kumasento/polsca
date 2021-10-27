#include <stdio.h>
#include <stdlib.h>

#define M 128
#define N 64

void kernel(int addrIn[M][N][N], int addrOut[M][N][N], double array[M * N],
            double A[M][N][N], double B[M][N][N], double C[M][N][N]) {
  double D[N][N], tmp[N][N];
  double beta = 0.7, alpha = 0.3;

  for (int p = 0; p < M; p++) {
    // hash
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        D[i][j] = array[addrIn[p][i][j]];

    // 2mm
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        tmp[i][j] = 0.0;
        for (int k = 0; k < N; ++k)
          tmp[i][j] += alpha * A[p][i][k] * B[p][k][j];
      }
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        D[i][j] *= beta;
        for (int k = 0; k < N; ++k)
          D[i][j] += tmp[i][k] * C[p][k][j];
      }

    // hash
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        array[addrOut[p][i][j]] = D[i][j];
  }
}

void initialize(int addrIn[M][N][N], int addrOut[M][N][N], double array[M * N],
                double A[M][N][N], double B[M][N][N], double C[M][N][N]) {
  for (int p = 0; p < M; p++) {
    for (int i = 0; i < N; i++) {
      array[p * N + i] = (double)rand() / RAND_MAX;
      for (int j = 0; j < N; j++) {
        addrIn[p][i][j] = rand() % (M * N);
        addrOut[p][i][j] = rand() % (M * N);
        A[p][i][j] = (double)rand() / RAND_MAX;
        B[p][i][j] = (double)rand() / RAND_MAX;
        C[p][i][j] = (double)rand() / RAND_MAX;
      }
    }
  }
}

int main() {
  // C99 syntax
  int(*addrIn)[M][N][N] = malloc(sizeof(int) * M * N * N);
  int(*addrOut)[M][N][N] = malloc(sizeof(int) * M * N * N);
  double(*array)[M * N] = malloc(sizeof(double) * M * N);
  double(*A)[M][N][N] = malloc(sizeof(double) * M * N * N);
  double(*B)[M][N][N] = malloc(sizeof(double) * M * N * N);
  double(*C)[M][N][N] = malloc(sizeof(double) * M * N * N);

  srand(9);

  initialize(*addrIn, *addrOut, *array, *A, *B, *C);
  kernel(*addrIn, *addrOut, *array, *A, *B, *C);

  for (int i = 0; i < M * N; i++)
    fprintf(stderr, "%.8lf\n", array[0][i]);

  return 0;
}
