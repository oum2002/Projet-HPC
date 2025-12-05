#ifndef MODEL_H
#define MODEL_H

#include <math.h>

// ===================================================
// ENUMÉRATIONS (Résout les erreurs 'undeclared')
// ===================================================

// Fonctions d'Activation supportées
typedef enum {
    ACT_TANH,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_LEAKY_RELU
} ActivationFn;

// Stratégies de Décroissance du Taux d'Apprentissage
typedef enum {
    DECAY_FIXED,
    DECAY_EXPONENTIAL,
    DECAY_INVERSE_TIME
} DecayStrategy;

// ----------------------
// Variables globales (Déclarées extern)
// ----------------------
extern int num_examples;    // Taille de l'ensemble d'entraînement
extern int nn_input_dim;
extern int nn_output_dim;
extern double reg_lambda;
extern double epsilon; 
extern double *X;          // Données d'entraînement
extern int *y;             // Labels d'entraînement

// Fonctions d'activation et leurs dérivés
double relu_derivative(double x);
double leaky_relu_derivative(double x);
double get_activation(double x, int activation_fn);

// Fonctions mathématiques du modèle (parallélisées)
void matmul(double *A, double *B, double *C, int M, int K, int N, int transpose_A); 
void add_bias(double *Z, double *b, int M, int N);
void softmax(double *z, double *probs, int M, int N);
double get_current_learning_rate(double initial_epsilon, double decay_coeff, int epoch, int strategy);

// Fonctions principales
double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim, int activation_fn);

// 1. Entraînement OpenMP (Tâches)
void build_model_minibatch(double *W1, double *b1, double *W2, double *b2, 
                           int nn_hdim, int num_passes, int batch_size, 
                           double initial_epsilon, double decay_coeff, int decay_strategy, 
                           int print_loss, int activation_fn);

// 2. Entraînement OpenMPI (Hybride)
void train_model_mpi(double *W1, double *b1, double *W2, double *b2, 
                     int nn_hdim, int num_passes, int batch_size, 
                     double initial_epsilon, double decay_coeff, int decay_strategy, 
                     int print_loss, int activation_fn,
                     double *X_local, int *y_local, int local_num_examples);


// Fonctions d'évaluation
void predict(double *W1, double *b1, double *W2, double *b2, int nn_hdim, double *X_data, int *y_pred, int current_num_examples, int activation_fn);
double calculate_accuracy(int *y_pred, int *y_true, int num_examples);

#endif
