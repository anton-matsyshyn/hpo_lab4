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
    int rows, columns, localColumns;
    double* matrix = NULL;
    double* vector = NULL;
    double* result = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Введення розмірності матриці та вектора на процесі 0
        printf("Введіть кількість рядків матриці: ");
        scanf("%d", &rows);

        printf("Введіть кількість стовпців матриці: ");
        scanf("%d", &columns);

        printf("Введіть значення вектора:\n");
        vector = (double*)malloc(columns * sizeof(double));
        for (int i = 0; i < columns; i++) {
            scanf("%lf", &vector[i]);
        }

        // Розсилка розмірності матриці та вектора на всі процеси
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Виділення пам'яті для матриці та результуючого вектора на процесі 0
        matrix = (double*)malloc(rows * columns * sizeof(double));
        result = (double*)malloc(rows * sizeof(double));

        // Введення значень матриці на процесі 0
        printf("Введіть значення матриці:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                scanf("%lf", &matrix[i * columns + j]);
            }
        }
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
