# method-
#Nurtay Moldir
```python
import numpy as np

# Обучающая выборка
X_train = np.array([[2, 4], [3, 6], [4, 4], [4, 6], [5, 2], [6, 4]])
y_train = np.array([0, 0, 1, 1, 2, 2])

# Новый объект для классификации
X_test = np.array([3.5, 5])

# Вычисление евклидовых расстояний
distances = np.linalg.norm(X_train - X_test, axis=1)

# Определение числа соседей
k = 3

# Индексы k ближайших соседей
nearest_neighbors = np.argsort(distances)[:k]

# Подсчет количества объектов каждого класса среди ближайших соседей
counts = np.bincount(y_train[nearest_neighbors])

# Прогнозируемая метка класса для нового объекта
predicted_class = np.argmax(counts)

print("Прогнозируемая метка класса:", predicted_class)
```
