<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Telegram-Animated-Emojis/main/Smileys/Alien%20Monster.webp" alt="Alien Monster" width="80" height="80" />

  <h1>🧵 Thread Programming with CUDA</h1>

  <p>
    <img src="https://img.shields.io/badge/CUDA-12.x-green" alt="CUDA">
    <img src="https://img.shields.io/badge/NVIDIA-GPU-yellow" alt="GPU">
    <img src="https://img.shields.io/badge/nvcc-Compiler-blue" alt="NVCC">
  </p>
</div>

---

### 💡 Overview
**Thread Programming with CUDA** es un conjunto de programas y ejemplos diseñados para entender y dominar el uso de **threads paralelos** en GPU usando **CUDA** de NVIDIA. Cada ejemplo demuestra cómo los hilos trabajan en paralelo para resolver tareas de manera eficiente.

Ideal para estudiantes, investigadores o desarrolladores que deseen explorar las bases del **cómputo paralelo** y la **programación en GPU**.

---

### 🚀 Getting Started

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/brivaro/cuda-thread-playground.git
   cd cuda-thread-playground
   ```

2. **Compila un archivo CUDA con `nvcc`:**
   ```bash
   nvcc example.cu -o example
   ```

   > ⚠️ Asegúrate de tener **CUDA Toolkit** instalado y que `nvcc` esté disponible en tu entorno.

3. **Ejecuta el binario:**
   ```bash
   ./example
   ```

---

### 🧠 ¿Qué encontrarás aquí?

- Ejemplos de **uso básico de hilos** (`threads`) en CUDA.  
- Demostraciones del modelo de **bloques e hilos** (`blocks` & `threads`).  
- Introducción al uso de **memoria compartida y sincronización**.  
- Casos prácticos: sumas paralelas, manipulación de arrays, matrices, etc.  

---

### 🛠 Tecnologías Utilizadas

- **CUDA con C** - Lenguaje para programación en GPU.  
- **NVCC** - Compilador de CUDA (parte del CUDA Toolkit).  
- **NVIDIA GPU** - Se requiere GPU compatible con CUDA para ejecución real.  

---

### 📌 Ejemplo de estructura de un kernel básico:

```cpp
__global__ void helloFromGPU() {
    printf("Hola desde el hilo %d del bloque %d\n", threadIdx.x, blockIdx.x);
}
```

---

¿Listo para desbloquear el poder de la **programación paralela** en GPU? 💥  
¡Explora, modifica y experimenta con los ejemplos!