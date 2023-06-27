#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void fillMatrixAndVector(double* matrix, double* vector, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i * columns + j] = (double)rand() / RAND_MAX;  // Випадкове значення від 0 до 1
        }
    }

    for (int i = 0; i < columns; i++) {
        vector[i] = (double)rand() / RAND_MAX;  // Випадкове значення від 0 до 1
    }
}

void matrixVectorMultiplication(double* matrix, double* vector, double* result, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < columns; j++) {
            result[i] += matrix[i * columns + j] * vector[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 9;
    int columns = 4;
    int localColumns = columns / size;
    int remainingColumns = columns % size;

    double* matrix = NULL;
    double* vector = NULL;
    double* result = NULL;
    double* localMatrix = NULL;
    double* localResult = NULL;

    if (rank == 0) {
        // Виділення пам'яті для матриці та вектора на процесі 0
        matrix = (double*)malloc(rows * columns * sizeof(double));
        vector = (double*)malloc(columns * sizeof(double));
        result = (double*)malloc(rows * sizeof(double));

        // Заповнення матриці та вектора випадковими значеннями
        srand(time(NULL));
        fillMatrixAndVector(matrix, vector, rows, columns);
    }

    // Розсилка кількості рядків та стовпців матриці
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Обчислення локального розміру стовпців для кожного процесу
    if (rank < remainingColumns) {
        localColumns += 1;
    }

    // Виділення пам'яті для локальних частин матриці та вектора
    localMatrix = (double*)malloc(rows * localColumns * sizeof(double));
    localResult = (double*)malloc(rows * sizeof(double));

    // Розсилка матриці та вектора з процесу 0 на всі процеси
    MPI_Bcast(vector, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix, rows * localColumns, MPI_DOUBLE, localMatrix, rows * localColumns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Виконання множення матриці на вектор
    matrixVectorMultiplication(localMatrix, vector, localResult, rows, localColumns);

    // Збір результатів на процесі 0
    MPI_Gather(localResult, rows, MPI_DOUBLE, result, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Виведення результату на процесі 0
        printf("Result:\n");
        for (int i = 0; i < rows; i++) {
            printf("%.2f ", result[i]);
        }
        printf("\n");

        // Звільнення пам'яті на процесі 0
        free(matrix);
        free(vector);
        free(result);
    }

    // Звільнення пам'яті на інших процесах
    free(localMatrix);
    free(localResult);

    MPI_Finalize();
    return 0;
}
