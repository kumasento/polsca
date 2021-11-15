#include <stdio.h>
#include <stdlib.h>

#define M 50
#define N 40
#define P 10
#define K 70
#define L 60

void kernel(int addrIn[P][M][L], int addrOut[P][M][L], double array[M * N],
            double A[P][M][K], double B[P][K][N], double C[P][N][L]) {
  double D[M][L], tmp[M][N];
  double beta = 0.7, alpha = 0.3;
  for (int p = 0; p < P; ++p) {
    // get hashed input
    for (int i = 0; i < M; ++i)
      for (int l = 0; l < L; ++l)
        D[i][l] = array[addrIn[p][i][l]];
    // two consecutive matrix multiplication
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        tmp[i][j] = 0.0;
        for (int k = 0; k < K; ++k)
          tmp[i][j] += alpha * A[p][i][k] * B[p][k][j];
      }
    for (int i = 0; i < M; ++i)
      for (int l = 0; l < L; ++l) {
        D[i][l] *= beta;
        for (int j = 0; j < N; ++j)
          D[i][l] += tmp[i][j] * C[p][j][l];
      }
    // get hashed output
    for (int i = 0; i < M; ++i)
      for (int l = 0; l < N; ++l)
        array[addrOut[p][i][l]] = D[i][l];
  }
}

void initialize(int addrIn[P][M][L], int addrOut[P][M][L], double array[M * N],
                double A[P][M][K], double B[P][K][N], double C[P][N][L]) {
  for (int i = 0; i < M * N; ++i)
    array[i] = (double)rand() / RAND_MAX;

  for (int p = 0; p < P; p++)
    for (int i = 0; i < M; i++)
      for (int l = 0; l < L; l++) {
        addrIn[p][i][l] = rand() % (M * N);
        addrOut[p][i][l] = rand() % (M * N);
      }

  for (int p = 0; p < P; ++p) {
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        A[p][i][k] = (double)rand() / RAND_MAX;

    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j)
        B[p][k][j] = (double)rand() / RAND_MAX;

    for (int j = 0; j < N; ++j)
      for (int l = 0; l < L; ++l)
        C[p][j][l] = (double)rand() / RAND_MAX;
  }
}

int main() {
  // C99 syntax
  int(*addrIn)[P][M][L] = malloc(sizeof(int) * P * M * L);
  int(*addrOut)[P][M][L] = malloc(sizeof(int) * P * M * L);
  double(*array)[M * N] = malloc(sizeof(double) * M * N);
  double(*A)[P][M][K] = malloc(sizeof(double) * P * M * K);
  double(*B)[P][K][N] = malloc(sizeof(double) * P * K * N);
  double(*C)[P][N][L] = malloc(sizeof(double) * P * N * L);

  srand(9);

  initialize(*addrIn, *addrOut, *array, *A, *B, *C);
  kernel(*addrIn, *addrOut, *array, *A, *B, *C);

  for (int i = 0; i < M * N; i++)
    fprintf(stderr, "%.8lf\n", array[0][i]);

  return 0;
}
