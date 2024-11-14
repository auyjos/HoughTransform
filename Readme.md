
---

# Proyecto Hough Transform en CUDA

Este proyecto implementa una transformación de Hough para detección de líneas en imágenes utilizando **CUDA** para procesamiento en paralelo y **OpenCV** para la carga y manipulación de imágenes.

## Descripción

El proyecto realiza la **Transformada de Hough** en imágenes, un algoritmo que permite detectar líneas rectas en una imagen. Se implementa en dos versiones:

- **CPU**: Implementación secuencial para validar los resultados.
- **GPU**: Implementación optimizada en CUDA para aprovechar el procesamiento paralelo en la tarjeta gráfica.

### Funcionalidad

1. **Transformada de Hough en CPU**: Calcula la acumulación de parámetros \(r\) y \(\theta\) para cada píxel de la imagen.
2. **Transformada de Hough en GPU**: Optimiza el cálculo de la acumulación utilizando memoria compartida y constante en la GPU.

### Requisitos

- **CUDA**: Para ejecutar el código en la GPU, es necesario tener un entorno compatible con CUDA.
- **OpenCV**: Para la carga y manipulación de imágenes.
- **g++**: Compilador para los archivos C++.
- **pkg-config**: Herramienta para obtener las configuraciones de OpenCV.

## Instrucciones de Instalación

### Dependencias

1. Instala **CUDA Toolkit** (versión compatible con tu GPU).
2. Instala **OpenCV** usando el siguiente comando en sistemas basados en Debian/Ubuntu:
   ```bash
   sudo apt-get install libopencv-dev
   ```
3. Asegúrate de tener **pkg-config** instalado para que el Makefile pueda obtener las configuraciones de OpenCV:
   ```bash
   sudo apt-get install pkg-config
   ```

### Compilación

1. Clona este repositorio en tu máquina.
2. Navega al directorio donde se encuentra el código y ejecuta:
   ```bash
   make
   ```
   Esto compilará el código CUDA y C++ y generará el ejecutable `houghGlobal` en el directorio `build`.

### Ejecución

Para ejecutar el programa, usa el siguiente comando:

```bash
./build/houghGlobal <ruta_de_imagen>
```

Sustituye `<ruta_de_imagen>` por el archivo de imagen que deseas procesar. La imagen debe estar en formato en escala de grises.

### Resultados

El programa guardará una imagen con las líneas detectadas en el directorio `images/` en formato `.png`.

## Estructura del Proyecto

```
.
├── build/                 # Directorio de salida para los archivos compilados
│   └── houghGlobal        # Ejecutable generado
├── common/                # Archivos comunes (por ejemplo, pgm.cpp)
│   └── pgm.cpp            # Funciones para cargar imágenes en formato PGM
├── src/                   # Código fuente principal
│   └── houghGlobal.cu     # Implementación de la Transformada de Hough en CUDA
├── images/                # Directorio donde se guardarán las imágenes de salida
├── Makefile               # Makefile para compilar el proyecto
```

## Explicación del Código

### Transformada de Hough en CPU

La función **CPU_HoughTran** realiza el procesamiento secuencial de la imagen para calcular la acumulación de parámetros \(r\) y \(\theta\) para cada píxel. La imagen se recorre píxel por píxel, y para cada píxel que no sea negro (valor mayor que 0), se calcula su contribución a la acumulación de la transformada en el espacio polar.

### Transformada de Hough en GPU

La función **GPU_HoughTran** optimiza el cálculo utilizando la paralelización en la GPU. Usa **memoria compartida** para almacenar los resultados intermedios y **memoria constante** para almacenar los valores de seno y coseno precalculados, lo que mejora la eficiencia en comparación con la versión de CPU.

### Makefile

El **Makefile** se encarga de compilar el proyecto y generar el ejecutable utilizando `nvcc` para el código CUDA y `g++` para el código C++. El Makefile también incluye una regla para limpiar los archivos generados.

## Limpieza de Archivos

Para eliminar los archivos generados y hacer una limpieza del proyecto, puedes ejecutar:

```bash
make clean
```

Este comando eliminará el ejecutable y los archivos objeto generados durante la compilación.

## Contribuciones

Si deseas contribuir a este proyecto, por favor haz un fork, realiza tus cambios y envía un pull request. Asegúrate de que tus cambios estén bien documentados y sean coherentes con el estilo del proyecto.

---