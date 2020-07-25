#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <fstream>
#include <time.h>

// Number of vertices
int N = 0;

__global__ void dijkstra(int *graph, int *result, bool *visited, int N) {

    // Init visited...
    int blockIndex1D = N * blockIdx.x;

    for (int vertex = 0; vertex < N; vertex++) {
        visited[blockIndex1D + vertex] = false;
        result[blockIndex1D + vertex] = INT_MAX;
    }

    // Distance from source to itself = 0
    result[blockIndex1D + blockIdx.x] = 0;

    for (int i = 0; i < N-1; i++) {
       // Get vertex with minimum distance
        int minDistance = INT_MAX;
        int minVertex;
        for (int vertex = 0; vertex < N; vertex++) {
            if (!visited[blockIndex1D + vertex] && result[blockIndex1D +  vertex] <= minDistance) {
                 minDistance = result[blockIndex1D + vertex];
                 minVertex = vertex;
            }
        }
 
        visited[blockIndex1D + minVertex] = true;
        int minBlockIndex1D = N * minVertex;
    
        for (int vertex = 0; vertex < N; vertex++) {
            if (!visited[blockIndex1D + vertex] &&
                 graph[minBlockIndex1D + vertex] &&
                 result[blockIndex1D + minVertex] != INT_MAX &&
                 result[blockIndex1D + minVertex] + graph[minBlockIndex1D + vertex] < result[blockIndex1D + vertex]) {
                     result[blockIndex1D + vertex] = result[blockIndex1D + minVertex] + graph[minBlockIndex1D + vertex];
                 }
        }
    }
}

int main(int argc, char *argv[]) {

    // Get matrix size from argument vector in , convert to int
    N = strtol(argv[1], NULL, 10);
    printf("N: %d\n ", N);

    int* dijkstraGraph, *result;
    cudaMallocManaged((void **) &dijkstraGraph, (sizeof(int) * N * N));
    cudaMallocManaged((void **) &result, (sizeof(int) * N * N));
    
    srand(13517115);
    // Fill the matrix with rand() function
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dijkstraGraph[i * N + j] = rand() % 1000;
            if (i == j) {
                dijkstraGraph[i * N + j] = 0;
            }
        }
    }

    bool *gpuVisited;
    cudaMalloc((void **) &gpuVisited, (sizeof(bool) * N * N));

    float totalTime = 0;
    clock_t start, end;
    start = clock();
    // Do the dijkstra: dimGrid = N, dimBlock = 1 (only 1 thread per block)
    dijkstra<<<N, 1>>>(dijkstraGraph, result, gpuVisited, N);
    cudaDeviceSynchronize();
    end = clock();
    totalTime = end - start;

    // Print elapsed time in microsecs
    printf("%f µs\n", totalTime);

    char filename[100];
    snprintf(filename, sizeof(char) * 100, "output-%i.txt", N);
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%d ", result[i * N + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    cudaFree(dijkstraGraph);
    cudaFree(result);

    return 0;
}
