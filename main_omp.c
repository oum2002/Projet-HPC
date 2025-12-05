#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <sys/stat.h>
#include <omp.h> 

// Définition des variables globales (utilisées par model.c)
int num_examples;
int nn_input_dim;
int nn_output_dim;
double reg_lambda = 0.01;
double epsilon = 0.01; 
double *X = NULL;
int *y = NULL;

// --- Helper Functions ---
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
}
// --- Fin des Helpers ---


int main(int argc, char *argv[]) {
    
    if(argc < 3) {
         fprintf(stderr, "Usage: %s <X_file> <y_file> [options]\n", argv[0]);
         fprintf(stderr, "Options:\n");
         fprintf(stderr, "  -t <n>, --threads <n> : Nombre de threads OpenMP (défaut: 1)\n");
         fprintf(stderr, "  ... (voir l'aide de mlp_mpi pour les autres options)\n");
         return 1;
    }
    
    const char *file_X = argv[1];
    const char *file_y = argv[2];

    nn_input_dim = 2;       
    nn_output_dim = 2;      
    int nn_hdim = 10;       

    // Paramètres
    int num_epochs = 20000;
    int batch_size_arg = 64;     
    double initial_epsilon = 0.01;
    double decay_coeff = 0.0;         
    int decay_strategy_arg = DECAY_FIXED; 
    int activation_fn_arg = ACT_LEAKY_RELU; 
    int num_threads_arg = 1; 
    double split_ratio = 0.8; 

    // Parse command line arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--activation") == 0 || strcmp(argv[i], "-a") == 0) {
            if (i + 1 < argc) {
                if (strcmp(argv[i+1], "relu") == 0) activation_fn_arg = ACT_RELU;
                else if (strcmp(argv[i+1], "tanh") == 0) activation_fn_arg = ACT_TANH;
                else if (strcmp(argv[i+1], "sigmoid") == 0) activation_fn_arg = ACT_SIGMOID;
                else if (strcmp(argv[i+1], "leaky_relu") == 0 || strcmp(argv[i+1], "lrelu") == 0) activation_fn_arg = ACT_LEAKY_RELU;
                i++;
            }
        } else if (strcmp(argv[i], "--epochs") == 0 || strcmp(argv[i], "-e") == 0) {
            if (i + 1 < argc) { num_epochs = atoi(argv[i + 1]); i++; }
        } else if (strcmp(argv[i], "--hidden") == 0 || strcmp(argv[i], "-h") == 0) {
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
        } else if (strcmp(argv[i], "--help") == 0) {
            // ... (Affichage de l'aide)
            return 0;
        }
    }

    if (decay_strategy_arg == DECAY_FIXED) decay_coeff = 0.0;
    
    // Définir le nombre de threads OpenMP
    omp_set_num_threads(num_threads_arg);

    // Compte tous les exemples et prépare le split
    int total_examples = count_lines(file_X);
    int num_train_examples = (int)(total_examples * split_ratio);
    int num_test_examples = total_examples - num_train_examples;

    // Ajuster batch_size pour qu'il soit un diviseur (simplifie la logique)
    if (num_train_examples % batch_size_arg != 0) {
        num_train_examples = (num_train_examples / batch_size_arg) * batch_size_arg;
        if(num_train_examples == 0) {
             fprintf(stderr, "Erreur: Le batch size (%d) est plus grand que le nombre d'exemples d'entraînement (%d)\n", 
                     batch_size_arg, (int)(total_examples * split_ratio));
             return 1;
        }
    }
    num_examples = num_train_examples; // Définit la globale pour model.c

    printf("========================================\n");
    printf("--- Configuration OpenMP Pur ---\n");
    printf("Total Samples: %d (Train: %d, Test: %d)\n", total_examples, num_train_examples, num_test_examples);
    printf("Hidden Dim: %d, Epochs: %d\n", nn_hdim, num_epochs);
    printf("Activation: %s\n", get_act_name(activation_fn_arg));
    printf("Mode: Mini-batch (size=%d)\n", batch_size_arg);
    printf("Learning Rate: %.4f (Strategy: %s, k=%.6f)\n", 
           initial_epsilon, get_decay_name(decay_strategy_arg), decay_coeff);
    printf("OpenMP Threads: %d\n", num_threads_arg);
    printf("========================================\n");


    // ========================================
    // 1. Chargement et Train/Test Split des données
    // ========================================
    double *X_all = malloc(total_examples * nn_input_dim * sizeof(double));
    int *y_all = malloc(total_examples * sizeof(int));
    load_X(file_X, X_all, total_examples, nn_input_dim);
    load_y(file_y, y_all, total_examples);
    
    // X et y sont les globales d'entraînement
    X = malloc(num_train_examples * nn_input_dim * sizeof(double));
    y = malloc(num_train_examples * sizeof(int));
    double *X_test = malloc(num_test_examples * nn_input_dim * sizeof(double));
    int *y_test = malloc(num_test_examples * sizeof(int));
    
    // Split des données
    memcpy(X, X_all, num_train_examples * nn_input_dim * sizeof(double));
    memcpy(y, y_all, num_train_examples * sizeof(int));
    memcpy(X_test, X_all + num_train_examples * nn_input_dim, num_test_examples * nn_input_dim * sizeof(double));
    memcpy(y_test, y_all + num_train_examples, num_test_examples * sizeof(int));
    free(X_all); free(y_all);
    
    // ========================================
    // 2. Allocation et Initialisation des poids
    // ========================================
    srand(0); 
    double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = calloc(nn_hdim, sizeof(double));
    double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim, sizeof(double));

    for (int i = 0; i < nn_input_dim * nn_hdim; i++) W1[i] = randn() / sqrt(nn_input_dim);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++) W2[i] = randn() / sqrt(nn_hdim);

    // ========================================
    // 3. Entraînement et Mesure du Temps
    // ========================================
    double start_time = omp_get_wtime();
    
    build_model_minibatch(W1, b1, W2, b2, nn_hdim, num_epochs, batch_size_arg, 
                          initial_epsilon, decay_coeff, decay_strategy_arg, 1, activation_fn_arg);
    
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("\nTraining Time: %.4f seconds\n", elapsed_time);

    // ========================================
    // 4. Evaluation Phase (on Test Set)
    // ========================================
    int *y_pred_test = malloc(num_test_examples * sizeof(int));

    printf("\nEvaluation on **Test** set...\n");
    predict(W1, b1, W2, b2, nn_hdim, X_test, y_pred_test, num_test_examples, activation_fn_arg);
    double test_accuracy = calculate_accuracy(y_pred_test, y_test, num_test_examples);
    
    printf("Final **Test** Accuracy: %.4f (Correct: %d/%d)\n", 
           test_accuracy, (int)(test_accuracy * num_test_examples), num_test_examples);
           
    // ========================================
    // 5. JSON Logging and Cleanup
    // ========================================
    log_benchmark_result("OpenMP", batch_size_arg, test_accuracy, num_epochs, nn_hdim, get_act_name(activation_fn_arg), decay_coeff, get_decay_name(decay_strategy_arg), num_threads_arg, 1, elapsed_time);
    
    // Libération des données et poids
    free(X); free(y); free(X_test); free(y_test);
    free(y_pred_test);
    free(W1); free(b1); free(W2); free(b2);
    
    printf("========================================\n");
    printf("Training and Evaluation completed!\n");
    printf("========================================\n");
    
    return 0;
}
