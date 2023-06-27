#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void matrixVectorMultiplication(double* matrix, double* vector, double* result, int rows, int columns, int localColumns) {
    // Виконуємо розсилку вектора на всі процеси
    MPI_Bcast(vector, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Кожен процес обчислює свій локальний результат
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < localColumns; j++) {
            result[i] += matrix[i * localColumns + j] * vector[j];
        }
    }

    // Збираємо локальні результати на процесі 0
    MPI_Reduce(result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int size, rank;
    int rows=9, columns=4, localColumns;
    double matrix[rows][colums]={
        {1, 2, -1, 2}, {3, 0, 4, -2}, {2, 3, 3, 5},
        {1, 2, -1, 2}, {2, 3, 3, 5}, {1, 2, -1, 2},
        {3, 0, 4, 3}, {1, 2, -1, 2}, {3, 0, 4,-2}
        };
    double vector[colums] = {-1, 2, 1, 3};
    double result[rows] = {0, };

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Розсилка розмірності матриці та вектора на всі процеси
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

        printf("Матриця \n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }

        printf("Вектор \n");

        for (int i = 0; i < rows; i++) {
            printf("%d ", vector[i][j]);
        }

        printf("\n");
    } else {
        // Отримання розмірності матриці та вектора на інших процесах
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Обчислення кількості стовпців, які будуть оброблятись на кожному процесі
    localColumns = columns / size;

    // Виділення пам'яті для локальних частин матриці та вектора
    double* localMatrix = (double*)malloc(rows * localColumns * sizeof(double));
    double* localResult = (double*)malloc(rows * sizeof(double));

    // Розсилка локальних частин матриці на всі процеси
    MPI_Scatter(matrix, rows * localColumns, MPI_DOUBLE, localMatrix, rows * localColumns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Виконання множення матриці на вектор
    matrixVectorMultiplication(localMatrix, vector, localResult, rows, columns, localColumns);

    // Збір результатів на процесі 0
    MPI_Gather(localResult, rows, MPI_DOUBLE, result, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Виведення результату на процесі 0
        printf("\nРезультат множення матриці на вектор:\n");
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