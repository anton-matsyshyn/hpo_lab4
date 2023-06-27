#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void matrixVectorMultiplication(double *matrix, double *vector, double *result, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < columns; j++) {
            result[i] += matrix[i * columns + j] * vector[j];
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int rows = 9; // Розмір матриці
    int columns = 4; // Розмір вектора
    int localRows; // Кількість рядків для кожного процесу
    int localColumns; // Кількість стовпців для кожного процесу
    double *matrix = NULL; // Матриця
    double *vector = NULL; // Вектор
    double *localMatrix = NULL; // Локальна матриця для кожного процесу
    double *localResult = NULL; // Локальний результат для кожного процесу
    double *result = NULL; // Результат

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Розрахунок кількості рядків і стовпців для кожного процесу
    localRows = rows / size;
    localColumns = columns;

    // Виділення пам'яті для локальних частин матриці та вектора
    localMatrix = (double *)malloc(localRows * localColumns * sizeof(double));
    localResult = (double *)malloc(localRows * sizeof(double));

    // Заповнення матриці та вектора якимись значеннями
    matrix = (double *)malloc(rows * columns * sizeof(double));
    vector = (double *)malloc(columns * sizeof(double));
    result = (double *)malloc(rows * sizeof(double));

    if (rank == 0) {
        // Ініціалізація матриці та вектора на процесі 0
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i * columns + j] = i + j + 1.0;
            }
        }

        for (int i = 0; i < columns; i++) {
            vector[i] = i + 1.0;
        }
    }

    // Розсилка матриці та вектора з процесу 0 на всі процеси
    MPI_Bcast(vector, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix, localRows * localColumns, MPI_DOUBLE, localMatrix, localRows * localColumns, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    // Виконання множення матриці на вектор на кожному процесі
    matrixVectorMultiplication(localMatrix, vector, localResult, localRows, localColumns);

    // Збір локальних результатів з усіх процесів на процесі 0
    MPI_Gather(localResult, localRows, MPI_DOUBLE, result, localRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Виведення результату на процесі 0
    if (rank == 0) {
        printf("\n");
        printf("Matrix:\n");
        for (int i=0; i<rows; i++) {
            for (int j=0; j<columns; j++) {
                printf("%.2f ", matrix[i * columns + j]);
            }
            printf("\n");
        }

        printf("\nVector:\n");
        for (int i=0; i<columns; i++) {
            printf("%.2f ", vector[i]);
        }

        printf("\n");
        printf("\nResult:\n");
        for (int i = 0; i < rows; i++) {
            printf("%.2f ", result[i]);
        }
        printf("\n");
    }

    // Звільнення пам'яті
    free(matrix);
    free(vector);
    free(localMatrix);
    free(localResult);
    free(result);

    MPI_Finalize();

    return 0;
}
