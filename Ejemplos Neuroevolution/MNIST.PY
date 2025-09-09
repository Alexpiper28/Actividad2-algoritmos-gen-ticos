import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# 1. Configuración
# ---------------------------
activaciones = ["relu", "tanh", "sigmoid", "elu", "gelu"]
tam_poblacion = 12      # población más grande
generaciones = 7        # más generaciones
epocas_entrenamiento = 10

# ---------------------------
# 2. Dataset MNIST
# ---------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255

# ---------------------------
# 3. Representación del cromosoma
# ---------------------------
def crear_cromosoma():
    num_capas = random.randint(1, 5)  # entre 1 y 5 capas
    neuronas = [random.choice([32, 64, 128, 256]) for _ in range(num_capas)]
    activacion = random.choice(activaciones)
    return [num_capas, neuronas, activacion]

# ---------------------------
# 4. Evaluación
# ---------------------------
def evaluar_cromosoma(cromosoma, guardar_mejor=False, nombre_modelo="mejor_modelo.h5"):
    num_capas, neuronas, activacion = cromosoma
    
    model = keras.Sequential()
    for n in neuronas:
        model.add(layers.Dense(n, activation=activacion))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    history = model.fit(
        x_train, y_train,
        epochs=epocas_entrenamiento,
        batch_size=128,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )
    
    acc = history.history["val_accuracy"][-1]

    if guardar_mejor:
        model.save(nombre_modelo)

    return acc

# ---------------------------
# 5. Selección
# ---------------------------
def seleccionar(poblacion, fitness, k=3):
    indices = np.argsort(fitness)[-k:]  # mejores k
    return [poblacion[i] for i in indices]

# ---------------------------
# 6. Cruzamiento
# ---------------------------
def cruzar(padre1, padre2):
    hijo = padre1.copy()
    if random.random() < 0.5:
        hijo[1] = padre2[1]
    if random.random() < 0.5:
        hijo[2] = padre2[2]
    return hijo

# ---------------------------
# 7. Mutación
# ---------------------------
def mutar(cromosoma):
    # mutar neuronas
    if random.random() < 0.3:
        cromosoma[1] = [random.choice([32, 64, 128, 256]) for _ in range(cromosoma[0])]
    # mutar activación
    if random.random() < 0.2:
        cromosoma[2] = random.choice(activaciones)
    # mutar número de capas
    if random.random() < 0.2:
        cromosoma[0] = random.randint(1, 5)
        cromosoma[1] = [random.choice([32, 64, 128, 256]) for _ in range(cromosoma[0])]
    return cromosoma

# ---------------------------
# 8. Algoritmo genético
# ---------------------------
poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
mejores_fitness = []
mejor_cromosoma_global = None
mejor_fitness_global = 0
contador_modelo = 0  # para guardar todos los modelos mejores

for gen in range(generaciones):
    print(f"\nGeneración {gen+1}")
    
    fitness = []
    for i, c in enumerate(poblacion):
        acc = evaluar_cromosoma(c)
        fitness.append(acc)
        print(f"Cromosoma {i}: {c} -> fitness={acc:.4f}")
    
    # Mejor de la generación
    idx_mejor = np.argmax(fitness)
    mejor = fitness[idx_mejor]
    mejores_fitness.append(mejor)
    print(f"👉 Mejor de generación {gen+1}: {mejor:.4f} con arquitectura {poblacion[idx_mejor]}")
    
    # Actualizar mejor global y guardar modelo
    if mejor > mejor_fitness_global:
        mejor_fitness_global = mejor
        mejor_cromosoma_global = poblacion[idx_mejor]
        contador_modelo += 1
        nombre_modelo = f"mejor_modelo_{contador_modelo}.h5"
        evaluar_cromosoma(mejor_cromosoma_global, guardar_mejor=True, nombre_modelo=nombre_modelo)
    
    # Selección
    mejores = seleccionar(poblacion, fitness, k=3)
    
    # Nueva población
    nueva_poblacion = mejores.copy()
    while len(nueva_poblacion) < tam_poblacion:
        p1, p2 = random.sample(mejores, 2)
        hijo = cruzar(p1, p2)
        hijo = mutar(hijo)
        nueva_poblacion.append(hijo)
    
    poblacion = nueva_poblacion

print("\nOptimización terminada ✅")
print(f"\n🏆 Mejor arquitectura encontrada: {mejor_cromosoma_global} con fitness={mejor_fitness_global:.4f}")

# ---------------------------
# 9. Evaluar mejor modelo guardado
# ---------------------------
print("\n🔎 Evaluando el mejor modelo en el set de test...")
best_model = keras.models.load_model(f"mejor_modelo_{contador_modelo}.h5")
loss, acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"📊 Accuracy final del mejor modelo en test: {acc:.4f}")

# ---------------------------
# 10. Graficar evolución del fitness
# ---------------------------
plt.plot(mejores_fitness, marker='o')
plt.xlabel("Generación")
plt.ylabel("Mejor fitness (accuracy)")
plt.title("Evolución del fitness con Algoritmo Genético")
plt.grid()
plt.show()
