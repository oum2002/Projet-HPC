#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include "model.h"
#include "utils.h"

// Définition des variables globales (utilisées par model.c)
double reg_lambda = 0.01;
int nn_input_dim = 0;
int nn_output_dim = 0;
int num_examples = 0; // Sera la taille TOTALE de l'entraînement
double *X = NULL; // Le processus 0 détient les données complètes
int *y = NULL;

// --- Helpers (copiés de main_omp.c pour l'affichage) ---
const char *get_decay_name(int decay_strategy) {
    if (decay_strategy == DECAY_EXPONENTIAL) return "Exponential";
    if (decay_strategy == DECAY_INVERSE_TIME) return "Inverse Time";
    return "Fixed"; 
}
const char *get_act_name(int activation_fn) {
    if (activation_fn == ACT_RELU) return "ReLU";
    if (activation_fn == ACT_SIGMOID) return "Sigmoid";
    if (activation_fn == ACT_LEAKY_RELU) return "Leaky ReLU";
    if (activation_fn == ACT_TANH) return "Tanh";
    return "Inconnu";
}
void log_benchmark_result(const char *mode, int batch_size, double accuracy, int num_epochs, int nn_hdim, const char *act_name, double decay_coeff, const char *decay_name, int num_threads, int mpi_size, double time_s) {
    const char *filename = "benchmark_results.json";
    FILE *fp = fopen(filename, "a"); 
    if (fp == NULL) {
        perror("Erreur: Impossible d'ouvrir le fichier JSON");
        return;
    }
    fprintf(fp, "{\"mode\": \"%s\", \"batch_size\": %d, \"accuracy\": %.6f, \"epochs\": %d, \"hidden_dim\": %d, \"activation\": \"%s\", \"decay_coeff\": %.6f, \"decay_schedule\": \"%s\", \"num_threads\": %d, \"mpi_size\": %d, \"time_s\": %.4f}\n",
        mode, batch_size, accuracy, num_epochs, nn_hdim, act_name, decay_coeff, decay_name, num_threads, mpi_size, time_s);
    fclose(fp);
    printf("\n[Rank 0] Benchmark result logged to %s\n", filename);
}
// --- Fin des Helpers ---


int main(int argc, char *argv[]) {
    // 1. Initialisation MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Paramètres
    int nn_hdim = 10;       
    int num_epochs = 20000;
    int batch_size_arg = 64;     
    double initial_epsilon = 0.01;
    double decay_coeff = 0.0;         
    int decay_strategy_arg = DECAY_FIXED; 
    int activation_fn_arg = ACT_LEAKY_RELU; 
    int num_threads_arg = 1; 
    double split_ratio = 0.8; 
    
    char *file_X = NULL;
    char *file_y = NULL;
    
    int num_train_examples = 0;
    int num_test_examples = 0;
    int total_examples = 0;

    // 2. Parsing des arguments (uniquement par le processus 0)
    if (rank == 0) {
        if(argc < 3) {
             fprintf(stderr, "Usage: mpirun -np <N> %s <X_file> <y_file> [options]\n", argv[0]);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        file_X = argv[1];
        file_y = argv[2];
        
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--activation") == 0 || strcmp(argv[i], "-a") == 0) {
                if (i + 1 < argc) {
                    if (strcmp(argv[i+1], "relu") == 0) activation_fn_arg = ACT_RELU;
                    else if (strcmp(argv[i+1], "tanh") == 0) activation_fn_arg = ACT_TANH;
                    else if (strcmp(argv[i+1], "sigmoid") == 0) activation_fn_arg = ACT_SIGMOID;
                    else if (strcmp(argv[i+1], "leaky_relu") == 0 || strcmp(argv[i+1], "lrelu") == 0) activation_fn_arg = ACT_LEAKY_RELU;
                    i++;
                }
            } else if (strcmp(argv[i], "--epochs") == 0 || strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "-passes") == 0) {
                if (i + 1 < argc) { num_epochs = atoi(argv[i + 1]); i++; }
            } else if (strcmp(argv[i], "--hidden") == 0 || strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-hdim") == 0) {
                if (i + 1 < argc) { nn_hdim = atoi(argv[i + 1]); i++; }
            } else if (strcmp(argv[i], "--lr") == 0) {
                if (i + 1 < argc) { initial_epsilon = atof(argv[i + 1]); i++; }
            } else if (strcmp(argv[i], "--decay") == 0 || strcmp(argv[i], "-d") == 0) {
                if (i + 1 < argc) { decay_coeff = atof(argv[i + 1]); i++; }
            } else if (strcmp(argv[i], "--schedule") == 0 || strcmp(argv[i], "-s") == 0) {
                if (i + 1 < argc) {
                    if (strcmp(argv[i+1], "exp") == 0) decay_strategy_arg = DECAY_EXPONENTIAL;
                    else if (strcmp(argv[i+1], "inv") == 0) decay_strategy_arg = DECAY_INVERSE_TIME;
                    else decay_strategy_arg = DECAY_FIXED;
                    i++;
                }
            } else if (strcmp(argv[i], "--minibatch") == 0 || strcmp(argv[i], "-mb") == 0) {
                if (i + 1 < argc) { batch_size_arg = atoi(argv[i + 1]); i++; }
            } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
                if (i + 1 < argc) {
                    num_threads_arg = atoi(argv[i + 1]);
                    if (num_threads_arg < 1) num_threads_arg = 1; 
                    i++;
                }
            }
        }
        
        // Détermination des dimensions
        nn_input_dim = 2; // Supposé
        nn_output_dim = 2;
        total_examples = count_lines(file_X);
        num_train_examples = (int)(total_examples * split_ratio);
        num_test_examples = total_examples - num_train_examples;
        
        // Ajustement pour MPI : num_train_examples doit être divisible par MPI_size
        if (num_train_examples % size != 0) {
            num_train_examples = (num_train_examples / size) * size;
            num_test_examples = total_examples - num_train_examples;
            
            printf("Warning: Train examples ajusté à %d pour être divisible par MPI_size (%d)\n", num_train_examples, size);
        }
        
        // CORRECTIF CRITIQUE : Mettre à jour num_examples (la variable globale)
        num_examples = num_train_examples;
    }
    
    // 3. Broadcast des paramètres
    MPI_Bcast(&nn_hdim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_epochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&batch_size_arg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&decay_coeff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reg_lambda, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&activation_fn_arg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&decay_strategy_arg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads_arg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nn_input_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nn_output_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_examples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_test_examples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Vérification de sécurité
    if (num_examples == 0) {
        if (rank == 0) {
            fprintf(stderr, "Erreur critique: num_examples est 0 après broadcast!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Définir le nombre de threads OpenMP
    omp_set_num_threads(num_threads_arg);
    
    // 4. Distribution des données
    int local_num_examples = num_examples / size;
    long local_data_size = (long)local_num_examples * nn_input_dim;
    double *X_local = (double *)malloc(local_data_size * sizeof(double));
    int *y_local = (int *)malloc(local_num_examples * sizeof(int));
    double *X_test = (double *)malloc(num_test_examples * nn_input_dim * sizeof(double));
    int *y_test = (int *)malloc(num_test_examples * sizeof(int));
    
    if (rank == 0) {
        X = (double *)malloc(num_examples * nn_input_dim * sizeof(double));
        y = (int *)malloc(num_examples * sizeof(int));
        
        // Charger TOUTES les données
        double *X_all = malloc(total_examples * nn_input_dim * sizeof(double));
        int *y_all = malloc(total_examples * sizeof(int));
        load_X(file_X, X_all, total_examples, nn_input_dim);
        load_y(file_y, y_all, total_examples);
        
        // Copier les données d'entraînement (X, y sont globales)
        memcpy(X, X_all, num_examples * nn_input_dim * sizeof(double));
        memcpy(y, y_all, num_examples * sizeof(int));
        
        // Copier les données de test (X_test, y_test locales)
        memcpy(X_test, X_all + num_examples * nn_input_dim, num_test_examples * nn_input_dim * sizeof(double));
        memcpy(y_test, y_all + num_examples, num_test_examples * sizeof(int));
        
        free(X_all); free(y_all);
        
        printf("========================================\n");
        printf("--- Configuration MPI/Hybride ---\n");
        printf("MPI Processes: %d, OMP Threads/Proc: %d (Total %d)\n", size, num_threads_arg, size * num_threads_arg);
        printf("Total Train Samples: %d, Local Samples/Proc: %d\n", num_examples, local_num_examples);
        printf("Activation: %s, Batch Size (OMP local): %d\n", get_act_name(activation_fn_arg), batch_size_arg);
        printf("Epochs: %d, Hidden Dim: %d\n", num_epochs, nn_hdim);
        printf("========================================\n");
    }
    
    // Scatter des données d'entraînement
    MPI_Scatter(X, local_data_size, MPI_DOUBLE, X_local, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, local_num_examples, MPI_INT, y_local, local_num_examples, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast des données de TEST (pour que Rank 0 puisse évaluer)
    MPI_Bcast(X_test, num_test_examples * nn_input_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_test, num_test_examples, MPI_INT, 0, MPI_COMM_WORLD);


    // 5. Initialisation et Broadcast des poids
    double *W1 = (double *)malloc((long)nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = (double *)calloc(nn_hdim, sizeof(double));
    double *W2 = (double *)malloc((long)nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = (double *)calloc(nn_output_dim, sizeof(double));

    if (rank == 0) {
        srand(0); 
        for (long i = 0; i < (long)nn_input_dim * nn_hdim; i++) W1[i] = randn() / sqrt(nn_input_dim);
        for (long i = 0; i < (long)nn_hdim * nn_output_dim; i++) W2[i] = randn() / sqrt(nn_hdim);
    }

    MPI_Bcast(W1, (long)nn_input_dim * nn_hdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b1, nn_hdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(W2, (long)nn_hdim * nn_output_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b2, nn_output_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 6. Entraînement (Chaque processus utilise ses données locales)
    double start_time = MPI_Wtime();

    // APPEL DE LA FONCTION D'ENTRAÎNEMENT MPI
    train_model_mpi(W1, b1, W2, b2, 
                    nn_hdim, num_epochs, batch_size_arg, 
                    initial_epsilon, decay_coeff, decay_strategy_arg, 
                    (rank == 0) ? 1 : 0,
                    activation_fn_arg,
                    X_local, y_local, local_num_examples);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // 7. Évaluation (par le Root)
    double test_accuracy = 0.0;
    if (rank == 0) {
        printf("\n--- Résultats MPI ---\n");
        printf("Temps total d'entraînement (MPI/Hybride): %.4f secondes\n", elapsed_time);
        
        // Évaluation sur l'ensemble de test
        int *y_pred_test = malloc(num_test_examples * sizeof(int));
        predict(W1, b1, W2, b2, nn_hdim, X_test, y_pred_test, num_test_examples, activation_fn_arg);
        test_accuracy = calculate_accuracy(y_pred_test, y_test, num_test_examples);
        
        printf("Final **Test** Accuracy: %.4f (Correct: %d/%d)\n", 
               test_accuracy, (int)(test_accuracy * num_test_examples), num_test_examples);
        
        // Logguer le résultat
        log_benchmark_result("MPI_Hybrid", batch_size_arg, test_accuracy, num_epochs, nn_hdim, get_act_name(activation_fn_arg), decay_coeff, get_decay_name(decay_strategy_arg), num_threads_arg, size, elapsed_time);
        
        printf("-----------------\n");
        free(y_pred_test);
    }
    
    // Libération de la mémoire
    if (rank == 0) {
        free(X); 
        free(y);
    }
    free(X_local);
    free(y_local);
    free(X_test);
    free(y_test);
    free(W1); free(b1); free(W2); free(b2);

    MPI_Finalize();
    return 0;
}
