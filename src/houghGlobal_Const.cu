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

// Kernel de GPU optimizado con memoria compartida y constante
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        // Usar memoria constante precargada para seno y coseno
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * c_Cos[tIdx] + yCoord * c_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins) {
                // Usamos atomicAdd para evitar condiciones de carrera
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

    // Imprimir las dimensiones de la imagen
    std::cout << "Tamaño de la imagen: " << w << "x" << h << std::endl;

    unsigned char *pic = new unsigned char[w * h];
    memcpy(pic, img.data, w * h);

    // Inicialización de memoria para seno y coseno
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Copiar valores precalculados a memoria constante
    cudaMemcpyToSymbol(c_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(c_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in;
    int *d_hough, *h_hough;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));
    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, pic, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Medición de tiempo usando eventos CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Definir un tamaño de bloque eficiente y la cantidad de bloques
    int threadsPerBlock = 256;
    int blockNum = (w * h + threadsPerBlock - 1) / threadsPerBlock;  // Redondeo hacia arriba

    GPU_HoughTran <<<blockNum, threadsPerBlock>>> (d_in, w, h, d_hough, rMax, rScale);
    
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiempo de ejecución del kernel: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Comprobación de umbral y dibujo de líneas en la imagen de salida
    cv::Mat output = img.clone();
    int threshold = 50;
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            if (h_hough[rIdx * degreeBins + tIdx] > threshold) {
                float theta = tIdx * radInc;
                float r = (rIdx * rScale) - rMax;
                cv::Point pt1, pt2;
                int x0 = r * cos(theta);
                int y0 = r * sin(theta);
                pt1.x = cvRound(x0 + 1000 * (-sin(theta)));
                pt1.y = cvRound(y0 + 1000 * cos(theta));
                pt2.x = cvRound(x0 - 1000 * (-sin(theta)));
                pt2.y = cvRound(y0 - 1000 * cos(theta));
                cv::line(output, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
        }
    }

    // Crear la carpeta "images" si no existe
    std::string outputDir = "images";
    std::filesystem::create_directory(outputDir); // Crea el directorio si no existe

    // Obtener el nombre base del archivo de entrada (sin la extensión)
    std::string inputFilePath = argv[1];
    std::string fileName = inputFilePath.substr(inputFilePath.find_last_of("/\\") + 1);
    std::string baseName = fileName.substr(0, fileName.find_last_of("."));
    
    // Crear el nombre del archivo de salida (con extensión .png)
    std::string outputFilePath = outputDir + "/" + baseName + "global_const.png";

    // Guardar la imagen de salida
    cv::imwrite(outputFilePath, output);

    std::cout << "Imagen guardada como: " << outputFilePath << std::endl;

    // Liberación de memoria
    delete[] pic;
    cudaFree(d_in);
    cudaFree(d_hough);
    free(h_hough);
    free(pcCos);
    free(pcSin);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
