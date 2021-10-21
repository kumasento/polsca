#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define N 16

void encrypt(int Sbox[M][N], int statemt[M]) {
  int ret[M];

  for (int i = 1; i <= 4; ++i) {
    for (int j = 0; j < N; j++)
      statemt[j * 4] = Sbox[statemt[j * 4] >> 4][statemt[j * 4] & 0xf];

    for (int j = 0; j < M - 1; ++j) {
      ret[j] = (statemt[j] << 1);
      if ((ret[j] >> 8) == 1)
        ret[j] ^= 283;
      int x = statemt[1 + j];
      x ^= (x << 1);
      if ((x >> 8) == 1)
        ret[j] ^= (x ^ 283);
      else
        ret[j] ^= x;
    }

    for (int j = 0; j < M; ++j)
      statemt[j] = ret[j];
  }
}

int main() {
  int Sbox[M][N], statemt[M];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      Sbox[i][j] = (i + j) % M;
    statemt[i] = i;
  }

  encrypt(Sbox, statemt);

  for (int i = 0; i < M; ++i)
    fprintf(stderr, "%08x\n", statemt[i]);

  return 0;
}
