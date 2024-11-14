#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include "../common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Variables en memoria constante para cosenos y senos
__constant__ float c_Cos[degreeBins];
__constant__ float c_Sin[degreeBins];

// Kernel utilizando Memoria Global
__global__ void GPU_HoughTran_Global(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *globalCos, float *globalSin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * globalCos[tIdx] + yCoord * globalSin[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
}

// Kernel utilizando Memoria Constante
__global__ void GPU_HoughTran_Const(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * c_Cos[tIdx] + yCoord * c_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
}

// Kernel utilizando Memoria Compartida
__global__ void GPU_HoughTran_Shared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    __shared__ float s_Cos[degreeBins];
    __shared__ float s_Sin[degreeBins];

    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < degreeBins) {
        s_Cos[threadIdx.x] = cosf(threadIdx.x * radInc);
        s_Sin[threadIdx.x] = sinf(threadIdx.x * radInc);
    }
    __syncthreads();  // Esperar a que todos los hilos carguen los valores de seno y coseno

    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * s_Cos[tIdx] + yCoord * s_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta de imagen>" << std::endl;
        return -1;
    }

    // Cargar la imagen con OpenCV
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    int w = img.cols;
    int h = img.rows;

    unsigned char *pic = new unsigned char[w * h];
    memcpy(pic, img.data, w * h);

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    unsigned char *d_in;
    int *d_hough, *h_hough;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));
    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, pic, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Inicialización de memoria global para seno y coseno
    float *globalCos = (float *)malloc(sizeof(float) * degreeBins);
    float *globalSin = (float *)malloc(sizeof(float) * degreeBins);
    for (int i = 0; i < degreeBins; i++) {
        globalCos[i] = cos(i * radInc);
        globalSin[i] = sin(i * radInc);
    }

    // Copiar valores de seno y coseno a memoria constante
    cudaMemcpyToSymbol(c_Cos, globalCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(c_Sin, globalSin, sizeof(float) * degreeBins);

    // Medición de tiempo usando eventos CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blockNum = (w * h + threadsPerBlock - 1) / threadsPerBlock;  // Redondeo hacia arriba

    // Pruebas de ejecución con cada kernel
    for (int i = 0; i < 3; i++) {
        cudaEventRecord(start);

        if (i == 0) {
            // Test con Memoria Global
            GPU_HoughTran_Global<<<blockNum, threadsPerBlock>>>(d_in, w, h, d_hough, rMax, rScale, globalCos, globalSin);
        } else if (i == 1) {
            // Test con Memoria Constante
            GPU_HoughTran_Const<<<blockNum, threadsPerBlock>>>(d_in, w, h, d_hough, rMax, rScale);
        } else {
            // Test con Memoria Compartida
            GPU_HoughTran_Shared<<<blockNum, threadsPerBlock>>>(d_in, w, h, d_hough, rMax, rScale);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Tiempo de ejecución con Kernel " << (i == 0 ? "Global" : (i == 1 ? "Const" : "Shared")) << ": " << milliseconds << " ms" << std::endl;
    }

    // Liberación de memoria
    delete[] pic;
    cudaFree(d_in);
    cudaFree(d_hough);
    free(h_hough);
    free(globalCos);
    free(globalSin);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
