# =========================================================
# Makefile pour MLP (OpenMPI et OpenMP Hybride/Pur)
# =========================================================

# Compilateurs
MPICC = mpicc
CC = gcc

# Fichiers sources (Communs)
SRCS_UTILS = utils.c
SRCS_MODEL = model.c

# Sources spécifiques
SRCS_MAIN_MPI = main_mpi.c
SRCS_MAIN_OMP = main_omp.c

# Exécutables
TARGET_MPI = mlp_mpi
TARGET_OMP = mlp_omp

# =========================================================
# Options de compilation
# =========================================================

# Options de base (Optimisation, Warnings, Debugging, Math rapide)
CFLAGS_BASE = -O3 -march=native -Wall -g -ffast-math

# Options pour le code C standard et MPI pur (pas de OpenMP)
CFLAGS_MPI = $(CFLAGS_BASE)

# Options pour le code contenant des directives OpenMP
CFLAGS_OMP = $(CFLAGS_BASE) -fopenmp

# Libraries de liaison
# -lm est nécessaire pour la librairie mathématique (exp, sqrt, log)
LDFLAGS = -lm

# =========================================================
# Cibles (Construction des deux)
# =========================================================

.PHONY: all omp mpi clean

# 'make all' ou 'make' construit les deux exécutables
all: $(TARGET_MPI) $(TARGET_OMP)

mpi: $(TARGET_MPI)
omp: $(TARGET_OMP)

# =========================================================
# Cible MPI / Hybride (mlp_mpi)
# =========================================================

# Fichiers objets pour la cible MPI
OBJS_MPI = main_mpi.o_mpi utils.o_mpi model.o_mpi

$(TARGET_MPI): $(OBJS_MPI)
	# Le linking pour MPI doit inclure -fopenmp car model.o_mpi l'utilise
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(MPICC) $(CFLAGS_OMP) $^ $(LDFLAGS) -fopenmp -o $@

main_mpi.o_mpi: main_mpi.c model.h utils.h
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(MPICC) $(CFLAGS_OMP) -c $< -o $@

utils.o_mpi: utils.c utils.h
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(MPICC) $(CFLAGS_MPI) -c $< -o $@

model.o_mpi: model.c model.h utils.h
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(MPICC) $(CFLAGS_OMP) -c $< -o $@

# =========================================================
# Cible OpenMP Pur (mlp_omp)
# =========================================================

# Fichiers objets pour la cible OpenMP Pure
OBJS_OMP = main_omp.o_omp utils.o_omp model.o_omp

$(TARGET_OMP): $(OBJS_OMP)
	# FIX APPLIQUÉ: Utilisation de MPICC pour le linking, car model.o_omp
	# contient des appels de fonction MPI (train_model_mpi) non résolus par GCC standard.
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(MPICC) $(CFLAGS_OMP) $^ $(LDFLAGS) -o $@

main_omp.o_omp: main_omp.c model.h utils.h
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(CC) $(CFLAGS_OMP) -c $< -o $@

utils.o_omp: utils.c utils.h
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(CC) $(CFLAGS_OMP) -c $< -o $@
	
model.o_omp: model.c model.h utils.h
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	$(CC) $(CFLAGS_OMP) -c $< -o $@

# =========================================================
# Nettoyage
# =========================================================

clean:
	# NOTE: L'indentation ci-dessous DOIT être un caractère TAB.
	rm -f $(TARGET_MPI) $(TARGET_OMP) *.o_mpi *.o_omp *.json
