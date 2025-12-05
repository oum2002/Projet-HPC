#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Variable pour randn (Méthode Box-Muller)
static double saved_normal_rv = 0.0;
static int has_saved = 0;

int count_lines(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur d'ouverture du fichier");
        exit(EXIT_FAILURE);
    }
    int count = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') {
            count++;
        }
    }
    // Gérer le cas où la dernière ligne n'a pas de \n (commun)
    if (count == 0 && ftell(file) > 0) count = 1;
    
    fclose(file);
    return count;
}


void load_X(const char *filename, double *X, int num_examples, int nn_input_dim) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur d'ouverture du fichier X");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_examples; i++) {
        for (int j = 0; j < nn_input_dim; j++) {
            if (fscanf(file, "%lf", &X[i * nn_input_dim + j]) != 1) {
                fprintf(stderr, "Erreur de lecture du fichier X à la ligne %d, colonne %d\n", i + 1, j + 1);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void load_y(const char *filename, int *y, int num_examples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Erreur d'ouverture du fichier y");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_examples; i++) {
        if (fscanf(file, "%d", &y[i]) != 1) {
            fprintf(stderr, "Erreur de lecture du fichier y à la ligne %d\n", i + 1);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

// Générateur de nombres aléatoires selon une distribution normale (Box-Muller)
double randn() {
    if (has_saved) {
        has_saved = 0;
        return saved_normal_rv;
    }

    double u1, u2, s;
    do {
        u1 = (double)rand() / RAND_MAX * 2.0 - 1.0;
        u2 = (double)rand() / RAND_MAX * 2.0 - 1.0;
        s = u1 * u1 + u2 * u2;
    } while (s >= 1.0 || s == 0.0);

    double mul = sqrt(-2.0 * log(s) / s);
    saved_normal_rv = u2 * mul;
    has_saved = 1;
    return u1 * mul;
}

// NEW: Implementation of shuffle_data using Fisher-Yates algorithm
void shuffle_data(double *X, int *y, int num_examples, int nn_input_dim) {
    for (int i = num_examples - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // Swap labels (y)
        int temp_y = y[i];
        y[i] = y[j];
        y[j] = temp_y;

        // Swap feature vectors (X)
        for (int k = 0; k < nn_input_dim; k++) {
            double temp_x = X[i * nn_input_dim + k];
            X[i * nn_input_dim + k] = X[j * nn_input_dim + k];
            X[j * nn_input_dim + k] = temp_x;
        }
    }
}
