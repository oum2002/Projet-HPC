#include "model.h"
#include "utils.h" 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h> // Inclus pour la compilation de la cible MPI

// Déclaration externe des variables globales (elles sont définies dans main_omp.c et main_mpi.c)
extern double reg_lambda;
extern int nn_input_dim;
extern int nn_output_dim;
extern int num_examples;
extern double *X;
extern int *y;


// ===================================================
// 1. Fonctions d'Activation et de Dérivation
// ===================================================

double relu(double x) { return (x > 0.0) ? x : 0.0; }
double relu_derivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double leaky_relu(double x) { return (x > 0.0) ? x : 0.01 * x; }
double leaky_relu_derivative(double x) { return (x > 0.0) ? 1.0 : 0.01; }

double get_activation(double x, int activation_fn) {
    if (activation_fn == ACT_TANH) return tanh(x);
    if (activation_fn == ACT_RELU) return relu(x);
    if (activation_fn == ACT_SIGMOID) return sigmoid(x);
    if (activation_fn == ACT_LEAKY_RELU) return leaky_relu(x);
    return x; 
}

// ===================================================
// 2. Fonctions Mathématiques de Base (Parallélisées)
// ===================================================

// Softmax
void softmax(double *z, double *probs, int M, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double max_val = z[i * N];
        for (int j = 1; j < N; j++) {
            if (z[i * N + j] > max_val) max_val = z[i * N + j];
        }
        double sum_exp = 0.0;
        for (int j = 0; j < N; j++) {
            probs[i * N + j] = exp(z[i * N + j] - max_val);
            sum_exp += probs[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            probs[i * N + j] /= sum_exp;
        }
    }
}

// Matmul (avec support A.T * B)
void matmul(double *A, double *B, double *C, int M, int K, int N, int transpose_A) {
    if (transpose_A) {
        // C(KxN) = A.T(KxM) * B(MxN)
        #pragma omp parallel for
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                double sum = 0.0;
                for (int m = 0; m < M; m++) {
                    sum += A[m * K + k] * B[m * N + n];
                }
                C[k * N + n] = sum;
            }
        }
    } else {
        // C(MxN) = A(MxK) * B(KxN)
        #pragma omp parallel for
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = sum;
            }
        }
    }
}

// Add Bias
void add_bias(double *Z, double *b, int M, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Z[i * N + j] += b[j];
        }
    }
}

// ===================================================
// 3. Helper pour le Learning Rate Decay (CORRIGÉ)
// ===================================================
double get_current_learning_rate(double initial_epsilon, double decay_coeff, int epoch, int strategy) {
    if (strategy == DECAY_EXPONENTIAL) {
        // Note: Utilisation de pow() qui est plus standard que exp(-k*t)
        return initial_epsilon * pow(decay_coeff, epoch); 
    } else if (strategy == DECAY_INVERSE_TIME) {
        return initial_epsilon / (1.0 + decay_coeff * epoch);
    } else {
        return initial_epsilon; // DECAY_FIXED
    }
}


// ===================================================
// 4. calculate_loss (Utilise l'activation)
// ===================================================
double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim, int activation_fn) {
    double *z1 = (double *)malloc(num_examples * nn_hdim * sizeof(double));
    double *a1 = (double *)malloc(num_examples * nn_hdim * sizeof(double));
    double *z2 = (double *)malloc(num_examples * nn_output_dim * sizeof(double));
    double *probs = (double *)malloc(num_examples * nn_output_dim * sizeof(double));
    if (!z1 || !a1 || !z2 || !probs) { perror("Allocation mémoire calculate_loss échouée"); exit(EXIT_FAILURE); }

    // Forward
    matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim, 0);
    add_bias(z1, b1, num_examples, nn_hdim);
    
    #pragma omp parallel for
    for (int i = 0; i < num_examples * nn_hdim; i++) a1[i] = get_activation(z1[i], activation_fn);
    
    matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim, 0);
    add_bias(z2, b2, num_examples, nn_output_dim);
    softmax(z2, probs, num_examples, nn_output_dim);

    // Loss
    double cross_entropy_loss = 0.0;
    #pragma omp parallel for reduction(+:cross_entropy_loss)
    for (int i = 0; i < num_examples; i++) {
        double p = probs[i * nn_output_dim + y[i]];
        if (p < 1e-12) p = 1e-12; 
        cross_entropy_loss -= log(p);
    }
    cross_entropy_loss /= num_examples;

    double reg_loss = 0.0;
    for (int i = 0; i < nn_input_dim * nn_hdim; i++) reg_loss += W1[i] * W1[i];
    for (int i = 0; i < nn_hdim * nn_output_dim; i++) reg_loss += W2[i] * W2[i];
    reg_loss = (reg_lambda / 2.0) * reg_loss;
    
    double loss = cross_entropy_loss + reg_loss;

    free(z1); free(a1); free(z2); free(probs);
    return loss;
}


// ===================================================
// 5. build_model_minibatch (OpenMP Tasks avec accumulation privée)
// ===================================================
void build_model_minibatch(double *W1, double *b1, double *W2, double *b2, 
                           int nn_hdim, int num_passes, int batch_size, 
                           double initial_epsilon, double decay_coeff, int decay_strategy, 
                           int print_loss, int activation_fn) {
    
    int num_batches = num_examples / batch_size; // Suppose num_examples est divisible
    int num_threads = omp_get_max_threads(); 
    
    // Tailles pour l'allocation
    long w1_size = (long)nn_input_dim * nn_hdim;
    long b1_size = (long)nn_hdim;
    long w2_size = (long)nn_hdim * nn_output_dim;
    long b2_size = (long)nn_output_dim;
    long z1_size = (long)batch_size * nn_hdim;
    long z2_size = (long)batch_size * nn_output_dim;

    // 1. Accumulateurs finaux pour l'époque (Partagés)
    double *dW1_accum_epoch = (double *)calloc(w1_size, sizeof(double));
    double *db1_accum_epoch = (double *)calloc(b1_size, sizeof(double));
    double *dW2_accum_epoch = (double *)calloc(w2_size, sizeof(double));
    double *db2_accum_epoch = (double *)calloc(b2_size, sizeof(double));
    
    // 2. Gradients Accumulators PRIVÉS PAR THREAD (pour la sommation)
    double *dW1_private = (double *)calloc(num_threads * w1_size, sizeof(double));
    double *db1_private = (double *)calloc(num_threads * b1_size, sizeof(double));
    double *dW2_private = (double *)calloc(num_threads * w2_size, sizeof(double));
    double *db2_private = (double *)calloc(num_threads * b2_size, sizeof(double));
    
    // 3. Scratch Space (Buffers) PRIVÉS PAR THREAD (pour le calcul)
    double *z1_scratch = (double *)calloc(num_threads * z1_size, sizeof(double));
    double *a1_scratch = (double *)calloc(num_threads * z1_size, sizeof(double));
    double *z2_scratch = (double *)calloc(num_threads * z2_size, sizeof(double));
    double *probs_scratch = (double *)calloc(num_threads * z2_size, sizeof(double));
    double *delta3_scratch = (double *)calloc(num_threads * z2_size, sizeof(double));
    double *delta2_scratch = (double *)calloc(num_threads * z1_size, sizeof(double));
    
    // 4. Buffers de gradients TEMPORAIRES (Privés par thread)
    double *dW2_batch_scratch = (double *)calloc(num_threads * w2_size, sizeof(double));
    double *db2_batch_scratch = (double *)calloc(num_threads * b2_size, sizeof(double));
    double *dW1_batch_scratch = (double *)calloc(num_threads * w1_size, sizeof(double));
    double *db1_batch_scratch = (double *)calloc(num_threads * b1_size, sizeof(double));


    if (!dW1_accum_epoch || !dW1_private || !z1_scratch || !dW1_batch_scratch) { 
        perror("Allocation mémoire massive OpenMP échouée"); 
        exit(EXIT_FAILURE); 
    }

    // --- Boucle d'entraînement ---
    for (int epoch = 0; epoch < num_passes; epoch++) {

        double current_epsilon = get_current_learning_rate(initial_epsilon, decay_coeff, epoch, decay_strategy);

        // Réinitialiser les accumulateurs (memset est rapide)
        memset(dW1_accum_epoch, 0, w1_size * sizeof(double));
        memset(db1_accum_epoch, 0, b1_size * sizeof(double));
        memset(dW2_accum_epoch, 0, w2_size * sizeof(double));
        memset(db2_accum_epoch, 0, b2_size * sizeof(double));
        
        memset(dW1_private, 0, num_threads * w1_size * sizeof(double));
        memset(db1_private, 0, num_threads * b1_size * sizeof(double));
        memset(dW2_private, 0, num_threads * w2_size * sizeof(double));
        memset(db2_private, 0, num_threads * b2_size * sizeof(double));
        
        if (print_loss && epoch % 1000 == 0) {
            // (Calcul de la perte omis pour l'instant pour accélérer l'entraînement)
        }
        
        // La fonction shuffle_data doit être déclarée dans model.h ou utils.h
        shuffle_data(X, y, num_examples, nn_input_dim);

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                for (int b = 0; b < num_batches; b++) {
                    
                    int offset = b * batch_size;
                    double *X_batch = &X[offset * nn_input_dim];
                    int *y_batch = &y[offset];

                    #pragma omp task firstprivate(batch_size, X_batch, y_batch)
                    {
                        int thread_id = omp_get_thread_num(); 
                        
                        // --- 1. Obtenir les pointeurs vers la mémoire privée de ce thread ---
                        
                        // Accumulateurs de gradients (privés)
                        double *dW1_thread = dW1_private + (long)thread_id * w1_size;
                        double *db1_thread = db1_private + (long)thread_id * b1_size;
                        double *dW2_thread = dW2_private + (long)thread_id * w2_size;
                        double *db2_thread = db2_private + (long)thread_id * b2_size;

                        // Scratch space (buffers de calcul)
                        double *z1 = z1_scratch + (long)thread_id * z1_size;
                        double *a1 = a1_scratch + (long)thread_id * z1_size;
                        double *z2 = z2_scratch + (long)thread_id * z2_size;
                        double *probs = probs_scratch + (long)thread_id * z2_size;
                        double *delta3 = delta3_scratch + (long)thread_id * z2_size;
                        double *delta2 = delta2_scratch + (long)thread_id * z1_size;
                        
                        // Buffers de gradients temporaires (locaux au batch)
                        double *dW2_batch = dW2_batch_scratch + (long)thread_id * w2_size;
                        double *db2_batch = db2_batch_scratch + (long)thread_id * b2_size;
                        double *dW1_batch = dW1_batch_scratch + (long)thread_id * w1_size;
                        double *db1_batch = db1_batch_scratch + (long)thread_id * b1_size;
                        
                        // RÀZ des buffers batch temporaires (importante car réutilisés)
                        memset(dW2_batch, 0, w2_size * sizeof(double));
                        memset(db2_batch, 0, b2_size * sizeof(double));
                        memset(dW1_batch, 0, w1_size * sizeof(double));
                        memset(db1_batch, 0, b1_size * sizeof(double));

                        // -------------------- 2. Forward/Backward (Calcul du gradient local) --------------------
                        
                        // Forward
                        matmul(X_batch, W1, z1, batch_size, nn_input_dim, nn_hdim, 0);
                        add_bias(z1, b1, batch_size, nn_hdim);
                        for (int i = 0; i < batch_size * nn_hdim; i++) a1[i] = get_activation(z1[i], activation_fn);
                        matmul(a1, W2, z2, batch_size, nn_hdim, nn_output_dim, 0);
                        add_bias(z2, b2, batch_size, nn_output_dim);
                        softmax(z2, probs, batch_size, nn_output_dim);

                        // Backward
                        memcpy(delta3, probs, batch_size * nn_output_dim * sizeof(double));
                        for (int i = 0; i < batch_size; i++) delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

                        // dW2_batch = A1.T @ Delta3 (Somme)
                        matmul(a1, delta3, dW2_batch, batch_size, nn_hdim, nn_output_dim, 1);
                        
                        // db2_batch = sum(Delta3)
                        for (int j = 0; j < nn_output_dim; j++) {
                            for (int i = 0; i < batch_size; i++) db2_batch[j] += delta3[i * nn_output_dim + j];
                        }
                        
                        // Delta2
                        double *W2_T = (double *)malloc(w2_size * sizeof(double)); 
                        for(int i=0; i<nn_hdim; i++)
                            for(int j=0; j<nn_output_dim; j++)
                                W2_T[j*nn_hdim + i] = W2[i*nn_output_dim + j];
                        matmul(delta3, W2_T, delta2, batch_size, nn_output_dim, nn_hdim, 0);
                        free(W2_T);
                        
                        for (int i = 0; i < batch_size * nn_hdim; i++) {
                            if (activation_fn == ACT_TANH) delta2[i] *= (1.0 - a1[i]*a1[i]);
                            else if (activation_fn == ACT_RELU) delta2[i] *= relu_derivative(z1[i]);
                            else if (activation_fn == ACT_SIGMOID) delta2[i] *= (a1[i] * (1.0 - a1[i]));
                            else if (activation_fn == ACT_LEAKY_RELU) delta2[i] *= leaky_relu_derivative(z1[i]);
                        }

                        // dW1_batch = X_batch.T @ Delta2 (Somme)
                        matmul(X_batch, delta2, dW1_batch, batch_size, nn_input_dim, nn_hdim, 1);
                        
                        // db1_batch = sum(Delta2)
                        for (int j = 0; j < nn_hdim; j++) {
                            for (int i = 0; i < batch_size; i++) db1_batch[j] += delta2[i * nn_hdim + j];
                        }

                        // -------------------- 3. Accumulation SANS CRITICAL --------------------
                        // (Accumulation dans la zone privée du thread)
                        for (long i = 0; i < w1_size; i++) dW1_thread[i] += dW1_batch[i];
                        for (long i = 0; i < b1_size; i++) db1_thread[i] += db1_batch[i];
                        for (long i = 0; i < w2_size; i++) dW2_thread[i] += dW2_batch[i];
                        for (long i = 0; i < b2_size; i++) db2_thread[i] += db2_batch[i];
                        
                    } // Fin du task
                } // Fin de la boucle de création de tâches
            } // Fin du single

            // -------------------- 4. Sommation EFFICACE des gradients privés --------------------
            #pragma omp taskwait 
            
            #pragma omp for nowait
            for(int tid = 0; tid < num_threads; tid++) {
                long offset_w1 = (long)tid * w1_size;
                long offset_b1 = (long)tid * b1_size;
                long offset_w2 = (long)tid * w2_size;
                long offset_b2 = (long)tid * b2_size;

                for (long i = 0; i < w1_size; i++) dW1_accum_epoch[i] += dW1_private[offset_w1 + i];
                for (long i = 0; i < b1_size; i++) db1_accum_epoch[i] += db1_private[offset_b1 + i];
                for (long i = 0; i < w2_size; i++) dW2_accum_epoch[i] += dW2_private[offset_w2 + i];
                for (long i = 0; i < b2_size; i++) db2_accum_epoch[i] += db2_private[offset_b2 + i];
            }
            
        } // Fin du #pragma omp parallel 

        // -------------------- 5. Mise à jour des paramètres (Séquentielle par Époque) --------------------
        double inv_num_examples = 1.0 / num_examples;

        for (long i = 0; i < w1_size; i++) {
            double avg_grad = dW1_accum_epoch[i] * inv_num_examples; 
            avg_grad += reg_lambda * W1[i]; 
            W1[i] -= current_epsilon * avg_grad;
        }
        for (long i = 0; i < b1_size; i++) {
            b1[i] -= current_epsilon * (db1_accum_epoch[i] * inv_num_examples);
        }
        for (long i = 0; i < w2_size; i++) {
            double avg_grad = dW2_accum_epoch[i] * inv_num_examples; 
            avg_grad += reg_lambda * W2[i]; 
            W2[i] -= current_epsilon * avg_grad;
        }
        for (long i = 0; i < b2_size; i++) {
            b2[i] -= current_epsilon * (db2_accum_epoch[i] * inv_num_examples);
        }
        
    } // Fin de l'epoch

    // =========================================================
    // 7. Libération de la mémoire
    // =========================================================
    free(dW1_accum_epoch); free(db1_accum_epoch); free(dW2_accum_epoch); free(db2_accum_epoch);
    free(dW1_private); free(db1_private); free(dW2_private); free(db2_private);
    
    free(z1_scratch); free(a1_scratch); free(z2_scratch); free(probs_scratch);
    free(delta3_scratch); free(delta2_scratch);
    free(dW2_batch_scratch); free(db2_batch_scratch); free(dW1_batch_scratch); free(db1_batch_scratch);
}

// ===================================================
// 6. train_model_mpi (Version OpenMPI/Hybride) - CORRIGÉ
// ===================================================

void train_model_mpi(double *W1, double *b1, double *W2, double *b2, 
                     int nn_hdim, int num_passes, int batch_size, 
                     double initial_epsilon, double decay_coeff, int decay_strategy, 
                     int print_loss, int activation_fn,
                     double *X_local, int *y_local, int local_num_examples) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tailles
    long w1_size = (long)nn_input_dim * nn_hdim;
    long b1_size = (long)nn_hdim;
    long w2_size = (long)nn_hdim * nn_output_dim;
    long b2_size = (long)nn_output_dim;
    
    // Buffers de gradients locaux (pour ce processus)
    double *dW1_local = (double *)calloc(w1_size, sizeof(double));
    double *db1_local = (double *)calloc(b1_size, sizeof(double));
    double *dW2_local = (double *)calloc(w2_size, sizeof(double));
    double *db2_local = (double *)calloc(b2_size, sizeof(double));
    
    // Buffers pour les gradients globaux (résultat de l'Allreduce)
    double *dW1_global = (double *)calloc(w1_size, sizeof(double));
    double *db1_global = (double *)calloc(b1_size, sizeof(double));
    double *dW2_global = (double *)calloc(w2_size, sizeof(double));
    double *db2_global = (double *)calloc(b2_size, sizeof(double));

    if (!dW1_local || !dW1_global) { 
        perror("Allocation mémoire MPI échouée"); 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Buffers temporaires (Taille du batch local = local_num_examples)
    double *z1 = (double *)malloc(local_num_examples * nn_hdim * sizeof(double));
    double *a1 = (double *)malloc(local_num_examples * nn_hdim * sizeof(double));
    double *z2 = (double *)malloc(local_num_examples * nn_output_dim * sizeof(double));
    double *probs = (double *)malloc(local_num_examples * nn_output_dim * sizeof(double));
    double *delta3 = (double *)malloc(local_num_examples * nn_output_dim * sizeof(double));
    double *delta2 = (double *)malloc(local_num_examples * nn_hdim * sizeof(double));
    
    // --- Boucle d'entraînement MPI ---
    for (int epoch = 0; epoch < num_passes; epoch++) {

        double current_epsilon = get_current_learning_rate(initial_epsilon, decay_coeff, epoch, decay_strategy);

        // Pas de memset(0) sur les globaux (ils reçoivent le Allreduce)
        memset(dW1_local, 0, w1_size * sizeof(double));
        memset(db1_local, 0, b1_size * sizeof(double));
        memset(dW2_local, 0, w2_size * sizeof(double));
        memset(db2_local, 0, b2_size * sizeof(double));
        
        double *X_batch = X_local;
        int *y_batch = y_local;
        
        // 1. Forward Propagation (MPI/OMP Hybride)
        matmul(X_batch, W1, z1, local_num_examples, nn_input_dim, nn_hdim, 0);
        add_bias(z1, b1, local_num_examples, nn_hdim);
        #pragma omp parallel for
        for (int i = 0; i < local_num_examples * nn_hdim; i++) a1[i] = get_activation(z1[i], activation_fn);
        
        matmul(a1, W2, z2, local_num_examples, nn_hdim, nn_output_dim, 0);
        add_bias(z2, b2, local_num_examples, nn_output_dim);
        softmax(z2, probs, local_num_examples, nn_output_dim);

        // 2. Backward Propagation (MPI/OMP Hybride)
        memcpy(delta3, probs, local_num_examples * nn_output_dim * sizeof(double));
        
        #pragma omp parallel for
        for (int i = 0; i < local_num_examples; i++) {
            delta3[i * nn_output_dim + y_batch[i]] -= 1.0;
        }
        
        // dW2 local = A1.T @ Delta3
        matmul(a1, delta3, dW2_local, local_num_examples, nn_hdim, nn_output_dim, 1);
        
        // db2 local: Somme
        #pragma omp parallel for
        for (int j = 0; j < nn_output_dim; j++) {
            for (int i = 0; i < local_num_examples; i++) {
                db2_local[j] += delta3[i * nn_output_dim + j];
            }
        }
        
        // Delta2
        double *W2_T = (double *)malloc(w2_size * sizeof(double)); 
        #pragma omp parallel for
        for(int i=0; i<nn_hdim; i++)
            for(int j=0; j<nn_output_dim; j++)
                W2_T[j*nn_hdim + i] = W2[i*nn_output_dim + j];
                
        matmul(delta3, W2_T, delta2, local_num_examples, nn_output_dim, nn_hdim, 0);
        free(W2_T);
        
        #pragma omp parallel for
        for (int i = 0; i < local_num_examples * nn_hdim; i++) {
            if (activation_fn == ACT_TANH) delta2[i] *= (1.0 - a1[i]*a1[i]);
            else if (activation_fn == ACT_RELU) delta2[i] *= relu_derivative(z1[i]);
            else if (activation_fn == ACT_SIGMOID) delta2[i] *= (a1[i] * (1.0 - a1[i]));
            else if (activation_fn == ACT_LEAKY_RELU) delta2[i] *= leaky_relu_derivative(z1[i]);
        }

        // dW1 local = X_batch.T @ Delta2
        matmul(X_batch, delta2, dW1_local, local_num_examples, nn_input_dim, nn_hdim, 1);
        
        // db1 local: Somme
        #pragma omp parallel for
        for (int j = 0; j < nn_hdim; j++) {
            for (int i = 0; i < local_num_examples; i++) {
                db1_local[j] += delta2[i * nn_hdim + j];
            }
        }
        
        // 3. Agrégation des gradients (Communication MPI)
        // Somme de toutes les sommes locales
        MPI_Allreduce(dW1_local, dW1_global, w1_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(db1_local, db1_global, b1_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(dW2_local, dW2_global, w2_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(db2_local, db2_global, b2_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // ============================================================
        // 4. Normalisation et Mise à jour des poids - CORRECTIF CRITIQUE
        // ============================================================
        // EXPLICATION:
        // - Chaque processus MPI calcule les gradients sur local_num_examples
        // - MPI_Allreduce avec MPI_SUM fait la SOMME de tous les gradients
        // - Après Allreduce, dW1_global contient la somme sur TOUS les exemples
        // - num_examples (variable globale) = nombre TOTAL d'exemples d'entraînement
        // 
        // ERREUR PRÉCÉDENTE: norm_factor = 1.0 / (num_examples * size)
        //   Cela divisait par (8000 * 4) = 32000 au lieu de 8000
        //   Le learning rate effectif était donc divisé par size!
        //
        // CORRECTION: norm_factor = 1.0 / num_examples
        //   Les gradients sont déjà agrégés sur tous les processus
        //   On normalise simplement par le nombre total d'exemples
        // ============================================================
        
        double norm_factor_global = 1.0 / (double)num_examples; 
        
        // Mise à jour de W2 et b2
        #pragma omp parallel for
        for (long i = 0; i < w2_size; i++) {
            double avg_grad = dW2_global[i] * norm_factor_global;
            W2[i] -= current_epsilon * (avg_grad + reg_lambda * W2[i]);
        }

        #pragma omp parallel for
        for (long i = 0; i < b2_size; i++) {
            b2[i] -= current_epsilon * db2_global[i] * norm_factor_global;
        }

        // Mise à jour de W1 et b1
        #pragma omp parallel for
        for (long i = 0; i < w1_size; i++) {
            double avg_grad = dW1_global[i] * norm_factor_global;
            W1[i] -= current_epsilon * (avg_grad + reg_lambda * W1[i]);
        }
        
        #pragma omp parallel for
        for (long i = 0; i < b1_size; i++) {
            b1[i] -= current_epsilon * db1_global[i] * norm_factor_global;
        }
    }
    
    free(dW1_local); free(db1_local); free(dW2_local); free(db2_local);
    free(dW1_global); free(db1_global); free(dW2_global); free(db2_global);
    free(z1); free(a1); free(z2); free(probs); free(delta3); free(delta2);
}

// ===================================================
// 7. predict (Evaluation)
// ===================================================
void predict(double *W1, double *b1, double *W2, double *b2, int nn_hdim, double *X_data, int *y_pred, int current_num_examples, int activation_fn) {
    double *z1 = (double *)malloc(current_num_examples * nn_hdim * sizeof(double));
    double *a1 = (double *)malloc(current_num_examples * nn_hdim * sizeof(double));
    double *z2 = (double *)malloc(current_num_examples * nn_output_dim * sizeof(double));
    double *probs = (double *)malloc(current_num_examples * nn_output_dim * sizeof(double));
    if (!z1 || !a1 || !z2 || !probs) { perror("Allocation mémoire predict échouée"); exit(EXIT_FAILURE); }

    // Forward pass
    matmul(X_data, W1, z1, current_num_examples, nn_input_dim, nn_hdim, 0);
    add_bias(z1, b1, current_num_examples, nn_hdim);
    #pragma omp parallel for
    for (int i = 0; i < current_num_examples * nn_hdim; i++) a1[i] = get_activation(z1[i], activation_fn);
    matmul(a1, W2, z2, current_num_examples, nn_hdim, nn_output_dim, 0);
    add_bias(z2, b2, current_num_examples, nn_output_dim);
    softmax(z2, probs, current_num_examples, nn_output_dim);

    // Déterminer la classe prédite (max probability)
    #pragma omp parallel for
    for (int i = 0; i < current_num_examples; i++) {
        int prediction = 0;
        double max_prob = -INFINITY;
        for (int k = 0; k < nn_output_dim; k++) {
            if (probs[i * nn_output_dim + k] > max_prob) {
                max_prob = probs[i * nn_output_dim + k];
                prediction = k;
            }
        }
        y_pred[i] = prediction;
    }
    
    free(z1); free(a1); free(z2); free(probs);
}

// ===================================================
// 8. calculate_accuracy
// ===================================================
double calculate_accuracy(int *y_pred, int *y_true, int num_examples) {
    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (int i = 0; i < num_examples; i++) {
        if (y_pred[i] == y_true[i]) {
            correct++;
        }
    }
    return (double)correct / num_examples;
}
