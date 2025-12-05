w#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <sys/stat.h>
#include <omp.h> // IMPORTANT

// --- Helper Functions ---

const char *get_decay_name(int decay_strategy) {
    if (decay_strategy == DECAY_EXPONENTIAL) return "Exponential (eta_0 * k^t)";
    if (decay_strategy == DECAY_INVERSE_TIME) return "Inverse Time (eta_0 / (1 + k*t))";
    return "Fixed (eta_0)"; 
}

const char *get_act_name(int activation_fn) {
    if (activation_fn == ACT_RELU) return "ReLU";
    if (activation_fn == ACT_SIGMOID) return "Sigmoid";
    if (activation_fn == ACT_LEAKY_RELU) return "Leaky ReLU";
    return "Tanh"; 
}

void log_benchmark_result(int batch_size, double accuracy, int num_epochs, int nn_hdim, const char *act_name, double decay_coeff, const char *decay_name, int num_threads, double time_s) {
    const char *filename = "benchmark_results.json";
    
    FILE *fp = fopen(filename, "a"); 
    if (fp == NULL) {
        perror("Erreur: Impossible d'ouvrir le fichier JSON");
        return;
    }

    // Ajout de 'num_threads' et 'time_s'
    fprintf(fp, "{\"batch_size\": %d, \"accuracy\": %.6f, \"epochs\": %d, \"hidden_dim\": %d, \"activation\": \"%s\", \"decay_coeff\": %.6f, \"decay_schedule\": \"%s\", \"num_threads\": %d, \"time_s\": %.4f}\n",
        batch_size, accuracy, num_epochs, nn_hdim, act_name, decay_coeff, decay_name, num_threads, time_s);
    
    fclose(fp);
    printf("\nBenchmark result logged to %s (NDJSON format)\n", filename);
}


// --- Main Function ---

int main(int argc, char *argv[]) {
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // Variables globales initialisées pour le chargement des données
    nn_input_dim = 2;       
    nn_output_dim = 2;      
    int nn_hdim = 10;       

    // Training parameters
    int num_epochs = 20000;
    int batch_size_arg = 64;     
    double initial_epsilon = 0.01;
    double decay_coeff = 0.0;         
    int decay_strategy_arg = DECAY_FIXED; 
    int activation_fn_arg = ACT_LEAKY_RELU; 
    
    // OpenMP parameter
    int num_threads_arg = 1; // Default to sequential

    // Data split configuration
    double split_ratio = 0.8; 

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
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
                if (strcmp(argv[i+1], "exp") == 0 || strcmp(argv[i+1], "exponential") == 0) decay_strategy_arg = DECAY_EXPONENTIAL;
                else if (strcmp(argv[i+1], "inv") == 0 || strcmp(argv[i+1], "inverse") == 0) decay_strategy_arg = DECAY_INVERSE_TIME;
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
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -a <func>, --activation <func>  Activation function (tanh, relu, sigmoid, leaky_relu/lrelu. Default: leaky_relu)\n");
            printf("  -e <n>, --epochs <n>            Number of epochs (default: 20000)\n");
            printf("  -h <n>, --hidden <n>            Hidden layer size (default: 10)\n");
            printf("  --lr <rate>                     Initial learning rate (e.g., 0.01). Default: 0.01\n");
            printf("  -s <type>, --schedule <type>    Decay schedule (exp, inv, fixed). Default: fixed\n");
            printf("  -d <coeff>, --decay <coeff>     Decay coefficient 'k' (for inv) or base (for exp). Default: 0.0\n");
            printf("  -mb <size>, --minibatch <size>  Mini-batch size (default: 64)\n");
            printf("  -t <n>, --threads <n>           Number of OpenMP threads (default: 1)\n");
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

    // Mise à jour de la variable globale num_examples pour les fonctions d'entraînement
    num_examples = num_train_examples; 

    printf("========================================\n");
    printf("Neural Network Configuration\n");
    printf("========================================\n");
    printf("Training Samples: %d (%.0f%%)\n", num_train_examples, split_ratio * 100);
    printf("Test Samples: %d (%.0f%%)\n", num_test_examples, (1.0 - split_ratio) * 100);
    printf("Hidden dimension: %d\n", nn_hdim);
    printf("Epochs: %d\n", num_epochs);
    printf("Activation function: %s\n", get_act_name(activation_fn_arg));
    printf("Training mode: Mini-batch (size=%d) with OpenMP Tasks\n", batch_size_arg);
    printf("Learning Rate: %.4f (Strategy: %s, Coeff k/base: %.6f)\n", 
           initial_epsilon, get_decay_name(decay_strategy_arg), decay_coeff);
    printf("OpenMP Threads: %d (Max available: %d)\n", num_threads_arg, omp_get_max_threads());
    printf("========================================\n");


    // ========================================
    // 1. Chargement et Train/Test Split des données
    // ========================================
    double *X_all = malloc(total_examples * nn_input_dim * sizeof(double));
    int *y_all = malloc(total_examples * sizeof(int));
    load_X(file_X, X_all, total_examples, nn_input_dim);
    load_y(file_y, y_all, total_examples);
    
    // Allocation des ensembles Train/Test (X et y SONT les variables globales d'entraînement)
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

    // Glorot/Xavier initialization
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
    log_benchmark_result(batch_size_arg, test_accuracy, num_epochs, nn_hdim, get_act_name(activation_fn_arg), decay_coeff, get_decay_name(decay_strategy_arg), num_threads_arg, elapsed_time);
    
    // Libération des données et poids
    free(X); free(y); free(X_test); free(y_test);
    free(y_pred_test);
    free(W1); free(b1); free(W2); free(b2);
    
    printf("========================================\n");
    printf("Training and Evaluation completed!\n");
    printf("========================================\n");
    
    return 0;
}
