#include <stdlib.h>
#define M 1024
#define N 16

void encrypt(int Sbox[M][N], int statement[M]) {
  int ret[M];

  for (int i = 1; i <= 4; ++i) {
    for (int j = 0; j < N; j++)
      statement[j * 4] = Sbox[statement[j * 4] >> 4][statement[j * 4] & 0xf];

    for (int j = 0; j < M - 1; ++j) {
      ret[j] = (statement[j] << 1);
      if ((ret[j] >> 8) == 1)
        ret[j] ^= 283;
      int x = statement[1 + j];
      x ^= (x << 1);
      if ((x >> 8) == 1)
        ret[j] ^= (x ^ 283);
      else
        ret[j] ^= x;
    }

    for (int j = 0; j < M; ++j)
      statement[j] = ret[j];
  }
}

int main() {
  int Sbox[M][N], statement[M], statement_[M];

  srand(0);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      Sbox[i][j] = rand();
    statement[i] = rand() % M;
    statement_[i] = statement[i];
  }

  { // Software model as reference
    int ret[M];

    for (int i = 1; i <= 4; ++i) {
      for (int j = 0; j < N; j++)
        statement[j * 4] = Sbox[statement[j * 4] >> 4][statement[j * 4] & 0xf];

      for (int j = 0; j < M - 1; ++j) {
        ret[j] = (statement[j] << 1);
        if ((ret[j] >> 8) == 1)
          ret[j] ^= 283;
        int x = statement[1 + j];
        x ^= (x << 1);
        if ((x >> 8) == 1)
          ret[j] ^= (x ^ 283);
        else
          ret[j] ^= x;
      }

      for (int j = 0; j < M; ++j)
        statement[j] = ret[j];
    }
  }

  encrypt(Sbox, statement_);

  int result = 0;
  for (int i = 0; i < M; i++)
    result += (statement[i] == statement_[i]);

  if (result == M)
    return 0;
  else
    return -1;
}
