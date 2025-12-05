#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h> // Pour srand, rand

// Constante pour pi (nécessaire pour randn)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Déclarations
int count_lines(const char *filename);
void load_X(const char *filename, double *X, int num_examples, int nn_input_dim);
void load_y(const char *filename, int *y, int num_examples);
double randn();

// NEW: Declaration for shuffle_data
void shuffle_data(double *X, int *y, int num_examples, int nn_input_dim);

#endif
