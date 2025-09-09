
# 1. Algoritmo Genético para Selección de Características

Este notebook implementa un **algoritmo genético** para realizar la **selección de características** utilizando el conjunto de datos **cáncer de mama** de `sklearn`. El objetivo es seleccionar las características más relevantes para predecir si un tumor es maligno o benigno. Un **árbol de decisión** es utilizado como clasificador y se emplea **validación cruzada** para evaluar la aptitud de las soluciones generadas por el algoritmo genético.

## Librerías Importadas

- **`random`**: Para manejar la aleatoriedad en el algoritmo genético.
- **`numpy`**: Para manejar eficientemente arrays y matrices.
- **`sklearn.datasets`**: Para cargar el dataset de cáncer de mama.
- **`sklearn.model_selection`**: Para aplicar validación cruzada con el clasificador.
- **`sklearn.tree`**: Para usar un clasificador de árbol de decisión.
- **`matplotlib.pyplot`**: Para graficar el progreso del algoritmo a través de las generaciones.

## Conjunto de Datos

El conjunto de datos de **cáncer de mama** contiene 30 características relacionadas con medidas de las células tumorales. El objetivo es predecir si el tumor es maligno o benigno. Las características incluyen el radio, la textura, el perímetro, el área, la suavidad, entre otras.

## Funcionamiento del Algoritmo Genético

- **Selección**: Se seleccionan individuos basados en su aptitud para formar la siguiente generación.
- **Cruzamiento (Crossover)**: Los individuos seleccionados intercambian partes de sus características para crear nuevas soluciones.
- **Mutación**: Se introducen cambios aleatorios para mantener la diversidad en la población.

## Evaluación

La aptitud de cada individuo se evalúa mediante un **árbol de decisión** entrenado con las características seleccionadas y validado mediante **validación cruzada**.

---

Este algoritmo genético permite identificar las mejores características para mejorar el rendimiento de un clasificador en tareas de clasificación, como la detección de tumores malignos en el cáncer de mama.


# 2. Optimización de Hiperparámetros con Algoritmos Genéticos (PyTorch + DEAP)

Este repositorio demuestra cómo optimizar hiperparámetros de un modelo sencillo de clasificación para MNIST usando un Algoritmo Genético (GA) implementado con DEAP y PyTorch.

🧭Objetivo
Buscar automáticamente una combinación de hiperparámetros (tasa de aprendizaje, tamaño de lote y optimizador) que maximice la precisión en el conjunto de prueba.

🧩Resumen de la solución
Modelo: Regresión logística (capa lineal + softmax implícito vía CrossEntropyLoss).

Datos: MNIST (60k train / 10k test), imágenes aplanadas a 784 características y escaladas a [0,1].

Dispositivo: Usa GPU (CUDA) si está disponible; si no, CPU.

GA (DEAP):

Individuo = [lr: float, batch_size_idx: int, optimizer_idx: int].
Espacio de búsqueda: lr ∈ [1e-4, 1e-1], batch_size ∈ {32, 64, 128, 256}, optimizer ∈ {Adam, SGD}.
Operadores: cruce de dos puntos, torneo para selección, mutación personalizada por gen.
Fitness: precisión en test después de entrenar por 20 épocas.
Salida: mejor individuo, sus hiperparámetros mapeados y precisión alcanzada; entrenamiento final extendido (100 épocas) con los mejores hiperparámetros.

🛠️ Requisitos
Python 3.9+ (recomendado)
PyTorch
torchvision
numpy
deap
Instalación
Crear y activar entorno (opcional)
python -m venv .venv
 Windows: .venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate

Instalar dependencias
pip install torch torchvision numpy deap
Nota: Instala la versión de torch/torchvision adecuada a tu sistema y CUDA. Consulta la guía oficial de PyTorch.

📁 Estructura del script
Carga y preparación de datos

Descarga MNIST con torchvision.datasets.MNIST.
Aplana imágenes a vectores de 784 y normaliza a [0,1].
Optimización: se mueven x_train, y_train, x_test, y_test una sola vez al dispositivo (GPU/CPU).
Modelo

SoftmaxModel: capa totalmente conectada Linear(784 → 10).
La CrossEntropyLoss aplica log_softmax internamente.
Función de aptitud (evaluate_hyperparameters)

Mapea genes del individuo a hiperparámetros reales.
Crea DataLoader según batch_size.
Entrena por 20 épocas.
Evalúa precisión en x_test/y_test. Devuelve una tupla (accuracy,) como exige DEAP.
Configuración DEAP

creator.FitnessMax(weights=(1.0,)) → maximizar precisión.
Individuo con 3 genes: lr, batch_size_idx, optimizer_idx.
Operadores: cxTwoPoint, selTournament(tournsize=3), mutate_hyperparameters(indpb=0.2).
Ejecución del GA

Parámetros por defecto: POP_SIZE=20, CXPB=0.5, MUTPB=0.2, NGEN=10.
algorithms.eaSimple con estadísticas (avg, std, min, max) y Hall of Fame.
Entrenamiento final

Reconstruye y entrena el modelo con los mejores hiperparámetros por 100 épocas.
Reporta precisión final en test.
▶️ Cómo ejecutar
Guarda el script como ga_hpo_mnist.py y ejecuta:

python ga_hpo_mnist.py
Salida esperada (aprox.):

Log del GA por generación con estadísticas.
Mejor individuo y su mapeo a hiperparámetros.
Entrenamiento final y precisión en test.
Tiempo de ejecución: depende del dispositivo (GPU vs CPU) y de POP_SIZE, NGEN y epochs (20 en evaluación, 100 en entrenamiento final).

⚙️ Hiperparámetros del GA
Tamaño de población (POP_SIZE): más grande → mejor exploración (mayor costo).
Probabilidad de cruce (CXPB): control del recombinado.
Probabilidad de mutación (MUTPB): evita estancamiento en óptimos locales.
Generaciones (NGEN): más iteraciones → convergencia más robusta.
Torneo (tournsize): presión selectiva; valores altos aceleran convergencia, pero pueden reducir diversidad.
Sugerencias:

Comienza con POP_SIZE entre 20–50 y NGEN entre 10–30.
Ajusta epochs de la evaluación (20) para equilibrar fidelidad del fitness vs tiempo.
🧪 Métrica de desempeño
Precisión en test.
Posibles extensiones: separar train/valid/test y usar precisión en validación como fitness para evitar sobreajuste a test.
🔁 Reproducibilidad
Para reproducibilidad, puedes fijar semillas:

import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
Opcional: comportamiento determinista (puede ralentizar)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
Inserta esto al inicio del script, tras las importaciones.

Cómo extender el espacio de búsqueda
1) Añadir nuevas opciones de batch_size
batch_size_options = [16, 32, 64, 128, 256, 512]
Y cambia los límites de random.randint para el gen del índice
toolbox.register("attr_batch_size", random.randint, 0, len(batch_size_options) - 1)
2) Incluir nuevas tasas de aprendizaje (mismo rango)
Ajusta los límites de random.uniform:

toolbox.register("attr_lr", random.uniform, 1e-5, 5e-1)
3) Probar más optimizadores
optimizer_options = ['Adam', 'SGD', 'RMSprop', 'AdamW']
Cambia límites del índice y la selección al crear el optimizador
4) Cambiar el modelo
Sustituye SoftmaxModel por una red más profunda:

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)
Recuerda mover el modelo a device y, si cambias la arquitectura, considera añadir épocas en la evaluación.

Buenas prácticas y notas de rendimiento
Mover tensores al dispositivo una sola vez (ya aplicado) reduce el overhead de copia.
Evita sobreajuste al conjunto de test: usa un split de validación dentro de evaluate_hyperparameters y reserva test solo para el final.
Parada temprana (early stopping) durante la evaluación puede acelerar el GA si no se observa mejora.
Penalización por complejidad: cuando uses modelos más grandes, puedes restar una pequeña penalización al fitness para favorecer modelos simples.
Caching (avanzado): guarda evaluaciones repetidas de individuos idénticos.
Solución de problemas (FAQ)
Q1. ModuleNotFoundError: No module named 'deap' Instala con pip install deap.

Q2. Se agota la memoria en GPU Reduce batch_size o usa CPU (CUDA no disponible) y/o disminuye POP_SIZE/NGEN.

Q3. El GA no mejora Incrementa MUTPB o CXPB, o baja tournsize para aumentar diversidad; amplía el rango de lr.

Q4. Ejecución muy lenta Baja epochs en la función de aptitud, reduce POP_SIZE/NGEN, o usa un modelo más pequeño.

Detalles de implementación clave del script
Individuo: lista con 3 genes [(float) lr, (int) batch_size_idx, (int) optimizer_idx].
Mutación por gen con probabilidad indpb=0.2.
Evaluación: entrenamiento in situ del modelo por 20 épocas y cálculo directo de precisión.
Hall of Fame (hof) guarda el mejor individuo global.
Estadísticas (DEAP tools.Statistics) para seguimiento del progreso por generación.

Ejemplo de salida (formato)
--- Iniciando la optimización de hiperparámetros con Algoritmo Genético ---
gen nevals  avg std min max
0   20  0.91    0.03    0.85    0.94
1   10  0.92    0.02    0.88    0.95
...
--- Optimización Finalizada ---
Mejor individuo encontrado: [0.0031, 2, 0]
Mejores hiperparámetros: {"learning_rate": 0.0031, "batch_size": 128, "optimizer": "Adam"}
Mejor precisión de validación durante la HPO: 0.9530
--- Entrenando modelo final con los mejores hiperparámetros ---
Época final [100/100], Pérdida: 0.3124
Precisión final en el conjunto de prueba: 0.9620
Los valores son ilustrativos; variarán según semillas, hardware y parámetros del GA.

Roadmap de mejoras
Añadir conjunto de validación dentro de evaluate_hyperparameters.
Soportar más hiperparámetros (momentum, weight decay, scheduler, capas ocultas, etc.).
Implementar paralelización de evaluaciones (DEAP + multiprocessing o joblib).
Incorporar early stopping y K-fold.
Guardar el mejor modelo y registros (CSV/JSON) de todas las evaluaciones.

Atribuciones
PyTorch
DEAP
MNIST dataset
Cita
Si usas este ejemplo en tu trabajo, puedes citarlo como:

"Optimización de hiperparámetros con Algoritmo Genético usando PyTorch y DEAP para MNIST (2025)".

# 3. Neuroevolution: Optimización de Redes Neuronales con Algoritmos Genéticos

Este repositorio contiene ejemplos de **Neuroevolution**, es decir, la optimización automática de arquitecturas de redes neuronales mediante **algoritmos genéticos**. El objetivo es encontrar la mejor arquitectura de red para clasificar dígitos del dataset MNIST.

---
## Ejemplo 1
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
```
### 2. Dataset
```
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255

-Se carga MNIST y se normalizan los píxeles entre 0 y 1.
-Se aplana cada imagen de 28x28 a un vector de 784 características.
```
### 3. Representación del Cromosoma
```
def crear_cromosoma():
    num_capas = random.randint(1, 5)
    neuronas = [random.choice([32, 64, 128, 256]) for _ in range(num_capas)]
    activacion = random.choice(activaciones)
    return [num_capas, neuronas, activacion]

-Cada cromosoma representa una arquitectura:
[número de capas, lista de neuronas por capa, función de activación].
```
### 4. Evaluación del Cromosoma
```
def evaluar_cromosoma(cromosoma, guardar_mejor=False, nombre_modelo="mejor_modelo.h5"):
    ...

-Se construye un modelo secuencial según el cromosoma.
-Se entrena brevemente (EarlyStopping incluido) y se obtiene la precisión de validación.
-Si es el mejor modelo hasta el momento, se guarda en disco (.h5).
```
### 5. Selección
```
def seleccionar(poblacion, fitness, k=3):
    indices = np.argsort(fitness)[-k:]  # mejores k
    return [poblacion[i] for i in indices]

-Se seleccionan las k mejores arquitecturas de la generación actual.
-Esto implementa elitismo, asegurando que los mejores cromosomas sobrevivan.
```
### 6. Cruzamiento
```
def cruzar(padre1, padre2):
    ...

-Combina dos cromosomas para generar un hijo:
-Puede intercambiar las capas de neuronas o la función de activación.
-Permite explorar nuevas combinaciones de arquitecturas.
```
### 7. Mutación
```
def mutar(cromosoma):
    ...

-Cambia aleatoriamente partes del cromosoma:
    -Número de neuronas por capa.
    -Función de activación.
    -Número de capas.
-Introduce diversidad genética, evitando convergencia prematura.
```
### 8. Algoritmo Genético
```
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
```
### 9. Evaluación Final
```
best_model = keras.models.load_model(f"mejor_modelo_{contador_modelo}.h5")
loss, acc = best_model.evaluate(x_test, y_test, verbose=0)

-Se carga el mejor modelo guardado.
-Se evalúa en el set de test para obtener la precisión final.
```
### 10. Visualización
```
plt.plot(mejores_fitness, marker='o')

-Grafica la evolución del fitness (accuracy) por generación.
-Permite observar cómo mejora la población a lo largo del tiempo.
```
### Resultados Obtenidos
```
-Mejor arquitectura: [2, [256], 'gelu']
-Fitness máximo (val_accuracy): 0.9824
-Accuracy en test: 0.9805
```

### Ejemplo 2:
```
C. Uso de algoritmos genéticos para hallar la mejor arquitectura de una red neuronal o “Neuroevolution”.

🧬 Optimización de Redes Neuronales con Algoritmo Genético (Titanic Dataset)
Este proyecto implementa un algoritmo genético (AG) para optimizar la arquitectura de una red neuronal profunda aplicada al famoso dataset del Titanic.
El objetivo es encontrar automáticamente la mejor combinación de capas y neuronas que maximicen la precisión en la tarea de predicción de supervivencia.

📂 Contenido del proyecto
AG_neuroevolution.py → Código principal con preprocesamiento, AG y entrenamiento de modelos.
evolucion_accuracy.png → Gráfico de la evolución de accuracy del mejor individuo por generación.
curva_accuracy_mejor_modelo.png → Curvas de accuracy (train vs val) del mejor modelo.
curva_loss_mejor_modelo.png → Curvas de pérdida (train vs val) del mejor modelo.
⚙️ Tecnologías utilizadas
Python 3.10+
TensorFlow/Keras → Construcción y entrenamiento de redes neuronales.
Scikit-learn → Preprocesamiento de datos.
Matplotlib → Visualización de métricas y evolución del AG.
NumPy & Pandas → Manejo de datos.
Algoritmo genético propio implementado desde cero.
📊 Dataset: Titanic
Se utiliza el dataset clásico de sobrevivientes del Titanic.
Puedes descargarlo desde Kaggle: Titanic Dataset
o usar cualquier versión CSV equivalente.

El dataset debe contener al menos las siguientes columnas:

Survived (etiqueta, 0 = no sobrevivió, 1 = sobrevivió)
Pclass (clase del pasajero)
Sex
Age
SibSp
Parch
Fare
Embarked
En el código, columnas irrelevantes como Name, Ticket, Cabin, PassengerId son eliminadas.

🧩 Estructura del algoritmo genético
Inicialización

Se crea una población de arquitecturas aleatorias.
Cada arquitectura está representada como una lista de enteros → número de neuronas por capa oculta.
Ejemplo: [32, 16] significa 2 capas ocultas con 32 y 16 neuronas respectivamente.
Fitness (Evaluación)

Cada arquitectura se transforma en un modelo Keras.
Se entrena con los datos de entrenamiento.
El fitness se mide como la accuracy de validación.
Selección

Se eligen los 2 mejores individuos de cada generación (elitismo).
Crossover (Reproducción)

Se combina parte de la arquitectura de un padre con parte de otro.
Mutación

Se altera aleatoriamente el número de neuronas en alguna capa.
Nueva población

Se construye una nueva generación con padres + hijos.
Iteración

El proceso se repite hasta alcanzar el número de generaciones definidas.
📈 Resultados
Se generan gráficos para interpretar los resultados:
Evolución del accuracy a lo largo de generaciones.
Curvas de entrenamiento (accuracy y pérdida) del mejor modelo.
Ejemplo de salida en consola:

=== Generación 1 ===Arquitectura: [42, 17] | Accuracy: 0.7989Arquitectura: [21] | Accuracy: 0.7854...Padres seleccionados: [[42, 17], [21]] === Mejor arquitectura encontrada ===Arquitectura: [42, 17] | Accuracy: 0.8021

🚀 Ejecución en Google Colab
Sube el dataset Titanic a tu Google Drive.
Monta el Drive en Colab:
from google.colab import drive
drive.mount('/content/drive')

3. Ajusta la ruta del dataset en el código:

titanic = pd.read_csv("/content/drive/MyDrive/Titanic-Dataset.csv")

4. Ejecuta todo el script.