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
    double matrix[rows][columns];
    double vector[rows];
    double result[columns];

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Розсилка розмірності матриці та вектора на всі процеси
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

        matrix[0][0]=1;
        matrix[0][1]=2;
        matrix[0][2]=-1;
        matrix[0][3]=2;

        matrix[1][0]=3;
        matrix[1][1]=0;
        matrix[1][2]=4;
        matrix[1][3]=-2;

        matrix[2][0]=5;
        matrix[2][1]=6;
        matrix[2][2]=-8;
        matrix[2][3]=1;

        matrix[3][0]=6;
        matrix[3][1]=3;
        matrix[3][2]=7;
        matrix[3][3]=-9;

        matrix[4][0]=4;
        matrix[4][1]=1;
        matrix[4][2]=-2;
        matrix[4][3]=6;

        matrix[5][0]=7;
        matrix[5][1]=5;
        matrix[5][2]=-8;
        matrix[5][3]=3;

        matrix[6][0]=5;
        matrix[6][1]=2;
        matrix[6][2]=-1;
        matrix[6][3]=6;

        matrix[7][0]=3;
        matrix[7][1]=7;
        matrix[7][2]=-9;
        matrix[7][3]=1;

        matrix[8][0]=3;
        matrix[8][1]=3;
        matrix[8][2]=-1;
        matrix[8][3]=4;

        vector[0]=-1;
        vector[1]=2;
        vector[2]=1;
        vector[3]=3;

        printf("Матриця \n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }

        printf("Вектор \n");

        for (int i = 0; i < rows; i++) {
            printf("%d ", vector[i]);
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
