# Definir el compilador y las rutas
NVCC = nvcc
CXX = g++

# Obtener las rutas y bibliotecas de OpenCV usando pkg-config
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Definir el archivo fuente
SRC_CXX = common/pgm.cpp
SRC_CUDA = src/houghGlobal.cu

# Definir el archivo objetivo (en el directorio build)
TARGET = build/houghGlobal

# Directorio de salida para los archivos objeto
BUILD_DIR = build

# Regla por defecto
all: $(BUILD_DIR) $(TARGET)

# Crear el directorio build si no existe
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compilar el archivo CUDA junto con el archivo C++ a un ejecutable
$(TARGET): $(SRC_CUDA) $(SRC_CXX)
	$(NVCC) -o $(TARGET) $(SRC_CUDA) $(SRC_CXX) $(OPENCV_FLAGS)

# Limpiar los archivos generados
clean:
	rm -f $(TARGET) $(BUILD_DIR)/*.o
