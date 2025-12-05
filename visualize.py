import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
JSON_FILE = 'benchmark_results.json'

def load_data():
    """Charge les résultats du benchmark depuis le fichier NDJSON ou utilise les données fournies."""
    
    # Données fournies par l'utilisateur
    test_data = [
        {"mode": "OpenMP", "batch_size": 64, "accuracy": 0.911000, "epochs": 20000, "hidden_dim": 10, "activation": "Leaky ReLU", "decay_coeff": 0.000000, "decay_schedule": "Fixed", "num_threads": 1, "mpi_size": 1, "time_s": 32.1592},
        {"mode": "OpenMP", "batch_size": 64, "accuracy": 0.876000, "epochs": 20000, "hidden_dim": 10, "activation": "Leaky ReLU", "decay_coeff": 0.000000, "decay_schedule": "Fixed", "num_threads": 4, "mpi_size": 1, "time_s": 17.4177},
        {"mode": "OpenMP", "batch_size": 64, "accuracy": 0.856500, "epochs": 20000, "hidden_dim": 10, "activation": "Leaky ReLU", "decay_coeff": 0.000000, "decay_schedule": "Fixed", "num_threads": 8, "mpi_size": 1, "time_s": 26.0551},
        {"mode": "MPI_Hybrid", "batch_size": 64, "accuracy": 0.911000, "epochs": 20000, "hidden_dim": 10, "activation": "Leaky ReLU", "decay_coeff": 0.000000, "decay_schedule": "Fixed", "num_threads": 4, "mpi_size": 2, "time_s": 17.6785},
        {"mode": "MPI_Hybrid", "batch_size": 64, "accuracy": 0.911000, "epochs": 20000, "hidden_dim": 10, "activation": "Leaky ReLU", "decay_coeff": 0.000000, "decay_schedule": "Fixed", "num_threads": 4, "mpi_size": 4, "time_s": 11.7614},
        {"mode": "MPI_Hybrid", "batch_size": 64, "accuracy": 0.911000, "epochs": 20000, "hidden_dim": 10, "activation": "Leaky ReLU", "decay_coeff": 0.000000, "decay_schedule": "Fixed", "num_threads": 4, "mpi_size": 8, "time_s": 28.6623}
    ]

    results = []
    # Tenter de charger le fichier, sinon utiliser les données de test
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Avertissement: Ligne corrompue ignorée: '{line}'")
    
    if not results and test_data:
        print(f"Avertissement: Le fichier {JSON_FILE} n'a pas été trouvé ou est vide. Utilisation des données de test fournies.")
        results = test_data
    
    if not results:
        print(f"Erreur: Aucune donnée de benchmark valide trouvée.")
        return None
    
    df = pd.DataFrame(results)
    
    # S'assurer que les colonnes clés sont numériques
    df['num_threads'] = pd.to_numeric(df['num_threads'], errors='coerce')
    df['mpi_size'] = pd.to_numeric(df['mpi_size'], errors='coerce')
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    
    df = df.dropna(subset=['num_threads', 'mpi_size', 'time_s', 'accuracy'])
    
    return df

def calculate_metrics(df):
    """Calcule les métriques de performance : Accélération."""
    
    # 1. Détermination de la Ligne de Base (1 thread, 1 MPI)
    baseline_run = df[(df['num_threads'] == 1) & (df['mpi_size'] == 1)]
    
    if baseline_run.empty:
        # Fallback au cas où le run 1T/1P n'existe pas
        T1 = df['time_s'].max() 
        print(f"Erreur: La ligne de base (1 thread, 1 MPI) n'a pas été trouvée. Accélération basée sur le temps max ({T1:.2f}s).")
    else:
        # Utiliser le temps du run séquentiel (1 thread, 1 MPI) comme T1
        T1 = baseline_run['time_s'].iloc[0]
        
    # 2. Calcul de l'Accélération (Speedup)
    df['speedup'] = T1 / df['time_s']
    
    return df, T1

def plot_triple_histogram(df, T1):
    """Génère une figure avec trois sous-graphiques en histogramme."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # 3 graphiques côte à côte
    plt.suptitle(f"Analyse des Benchmarks Parallèles (Histogrammes) - Réf. $T_1={T1:.2f}$ s", fontsize=18, y=1.04)
    
    # --- Sous-Graphique 1: Mise à l'Échelle OpenMP (num_threads vs Speedup) ---
    df_omp = df[(df['mode'] == 'OpenMP') & (df['mpi_size'] == 1)]
    ax1 = axes[0]
    
    # S'assurer que le run de base est inclus, même si son Accélération est 1.0
    if df_omp.empty:
        ax1.text(0.5, 0.5, 'Aucune donnée OpenMP', transform=ax1.transAxes, ha='center')
    else:
        x_labels_omp = df_omp['num_threads'].apply(lambda x: f"T={x}")
        bars1 = ax1.bar(x_labels_omp, df_omp['speedup'], color='#1f77b4', alpha=0.8)
        
        # Accélération Idéale (égale au nombre de threads)
        ideal_omp = df_omp['num_threads']
        ax1.plot(x_labels_omp, ideal_omp, linestyle='--', color='red', marker='o', label='Idéale (N)')

        ax1.set_title('1. Mise à l\'Échelle OpenMP (Threads)', fontsize=14)
        ax1.set_xlabel('Nombre de Threads', fontsize=12)
        ax1.set_ylabel('Accélération ($S = T_1 / T_N$)', fontsize=12)
        ax1.legend()
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}x', ha='center', va='bottom', fontsize=10)


    # --- Sous-Graphique 2: Mise à l'Échelle MPI (mpi_size vs Speedup) ---
    # Filtrer les runs MPI/Hybrides (avec num_threads > 1, mais constante)
    df_mpi = df[(df['mode'] == 'MPI_Hybrid') & (df['num_threads'] > 1)]
    ax2 = axes[1]
    
    if df_mpi.empty:
        ax2.text(0.5, 0.5, 'Aucune donnée MPI/Hybride', transform=ax2.transAxes, ha='center')
    else:
        # Prendre le nombre de threads constant (pour le label)
        constant_threads = df_mpi['num_threads'].iloc[0]
        x_labels_mpi = df_mpi['mpi_size'].apply(lambda x: f"P={x}")
        bars2 = ax2.bar(x_labels_mpi, df_mpi['speedup'], color='#2ca02c', alpha=0.8)
        
        # Accélération Idéale (égale au nombre total de cœurs (P * constant_threads))
        ideal_mpi = df_mpi['mpi_size'] * constant_threads
        ax2.plot(x_labels_mpi, ideal_mpi, linestyle='--', color='red', marker='o', label=f'Idéale (P * {constant_threads})')
        
        ax2.set_title(f'2. Mise à l\'Échelle MPI/Hybride (T={constant_threads} fixe)', fontsize=14)
        ax2.set_xlabel('Nombre de Processus MPI', fontsize=12)
        ax2.set_ylabel('Accélération ($S = T_1 / T_N$)', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}x', ha='center', va='bottom', fontsize=10)


    # --- Sous-Graphique 3: Comparaison OpenMP vs MPI_Hybrid (Meilleurs Runs) ---
    
    # Trouver le meilleur run OpenMP pur (max speedup)
    best_omp_run = df_omp.sort_values(by='speedup', ascending=False).iloc[0]
    
    # Trouver le meilleur run MPI/Hybride (max speedup)
    best_mpi_run = df_mpi.sort_values(by='speedup', ascending=False).iloc[0]
    
    comparison_data = pd.DataFrame({
        'Mode': [
            f"OpenMP\n(T={best_omp_run['num_threads']})",
            f"MPI/Hybride\n(T={best_mpi_run['num_threads']}, P={best_mpi_run['mpi_size']})"
        ],
        'Speedup': [best_omp_run['speedup'], best_mpi_run['speedup']],
        'Accuracy': [best_omp_run['accuracy'], best_mpi_run['accuracy']]
    })
    
    ax3 = axes[2]
    # Position des barres pour les deux métriques
    x = np.arange(len(comparison_data['Mode']))
    width = 0.35
    
    bars3_speedup = ax3.bar(x - width/2, comparison_data['Speedup'], width, label='Accélération', color='#4c72b0', alpha=0.8)
    bars3_accuracy = ax3.bar(x + width/2, comparison_data['Accuracy'], width, label='Précision', color='#c44e52', alpha=0.8)

    ax3.set_title('3. Accélération et Précision: OpenMP vs Hybride', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(comparison_data['Mode'])
    ax3.set_xlabel('Mode de Parallélisation', fontsize=12)
    ax3.set_ylabel('Valeur (Accélération / Précision)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Affichage des valeurs sur les barres
    for bar in bars3_speedup:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}x', ha='center', va='bottom', fontsize=10, color='#4c72b0')
        
    for bar in bars3_accuracy:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, color='#c44e52')
        
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarde
    output_filename = 'triple_parallel_histogram_analysis.png'
    plt.savefig(output_filename) 
    print(f"\n[SUCCESS] Graphique d'analyse en triple histogramme généré: '{output_filename}'")
    plt.show()

def main():
    """Fonction principale pour charger les données, calculer les métriques et générer les graphiques."""
    df = load_data()
    
    if df is None:
        return
        
    df_metrics, T1 = calculate_metrics(df)
    
    plot_triple_histogram(df_metrics, T1)
    
if __name__ == '__main__':
    main()
