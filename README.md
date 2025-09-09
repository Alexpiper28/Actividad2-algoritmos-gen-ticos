
# Neuroevolution: Optimización de Redes Neuronales con Algoritmos Genéticos

Este repositorio contiene ejemplos de **Neuroevolution**, es decir, la optimización automática de arquitecturas de redes neuronales mediante **algoritmos genéticos**. El objetivo es encontrar la mejor arquitectura de red para clasificar dígitos del dataset MNIST.

---

## Contenido del Ejemplo 1 de Neuroevolution

- `MNIST.py` : Script principal que implementa el algoritmo genético para hallar la mejor arquitectura.  
- `mejor_modelo_5.h5` : Modelos guardados durante el entrenamiento (uno por mejora encontrada).  
- `README.md` : Documentación y guía de ejecución del proyecto.

---

## Funcionamiento del Código

El código está organizado en varias secciones, cada una con un propósito específico:

### 1. Configuración

```python
activaciones = ["relu", "tanh", "sigmoid", "elu", "gelu"]
tam_poblacion = 12
generaciones = 7
epocas_entrenamiento = 10

    -Define las funciones de activación posibles.
    -Configura el tamaño de la población, el número de generaciones y las épocas de entrenamiento de cada red.

### 2. Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255

-Se carga MNIST y se normalizan los píxeles entre 0 y 1.
-Se aplana cada imagen de 28x28 a un vector de 784 características.

### 3. Representación del Cromosoma
def crear_cromosoma():
    num_capas = random.randint(1, 5)
    neuronas = [random.choice([32, 64, 128, 256]) for _ in range(num_capas)]
    activacion = random.choice(activaciones)
    return [num_capas, neuronas, activacion]

-Cada cromosoma representa una arquitectura:
[número de capas, lista de neuronas por capa, función de activación].

### 4. Evaluación del Cromosoma
def evaluar_cromosoma(cromosoma, guardar_mejor=False, nombre_modelo="mejor_modelo.h5"):
    ...

-Se construye un modelo secuencial según el cromosoma.
-Se entrena brevemente (EarlyStopping incluido) y se obtiene la precisión de validación.
-Si es el mejor modelo hasta el momento, se guarda en disco (.h5).

### 5. Selección
def seleccionar(poblacion, fitness, k=3):
    indices = np.argsort(fitness)[-k:]  # mejores k
    return [poblacion[i] for i in indices]

-Se seleccionan las k mejores arquitecturas de la generación actual.
-Esto implementa elitismo, asegurando que los mejores cromosomas sobrevivan.

### 6. Cruzamiento
def cruzar(padre1, padre2):
    ...

-Combina dos cromosomas para generar un hijo:
-Puede intercambiar las capas de neuronas o la función de activación.
-Permite explorar nuevas combinaciones de arquitecturas.

### 7. Mutación
def mutar(cromosoma):
    ...

-Cambia aleatoriamente partes del cromosoma:
    -Número de neuronas por capa.
    -Función de activación.
    -Número de capas.
-Introduce diversidad genética, evitando convergencia prematura.

### 8. Algoritmo Genético
poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
...

Se inicializa una población de cromosomas aleatorios.
Por cada generación:
    1. Se evalúa cada cromosoma y se guarda su fitness.
    2. Se identifica el mejor cromosoma de la generación.
    3. Se actualiza el mejor global y se guarda el modelo.
    4. Se seleccionan los mejores para reproducirse.
    5. Se generan nuevos cromosomas mediante cruzamiento y mutación.
Se repite durante todas las generaciones, optimizando la arquitectura.

### 9. Evaluación Final
best_model = keras.models.load_model(f"mejor_modelo_{contador_modelo}.h5")
loss, acc = best_model.evaluate(x_test, y_test, verbose=0)

-Se carga el mejor modelo guardado.
-Se evalúa en el set de test para obtener la precisión final.

### 10. Visualización
plt.plot(mejores_fitness, marker='o')

-Grafica la evolución del fitness (accuracy) por generación.
-Permite observar cómo mejora la población a lo largo del tiempo.

'Resultados Obtenidos'
-Mejor arquitectura: [2, [256], 'gelu']
-Fitness máximo (val_accuracy): 0.9824
-Accuracy en test: 0.9805