import numpy as np
from main import NeuralNetwork


X = np.array([
    [1, 3],
    [2, 4], 
    [1, 5]
])

y = np.array([8, 11, 9])

nn = NeuralNetwork(input_size=2, neurons_count=1)

print("Веса до обучения:", nn.weights)
nn.fit(X, y, epochs=100)
print("Веса после обучения:", nn.weights)

print("\nТестирование:")
for i in range(len(X)):
    prediction = nn.predict(X[i])
    print(f"Вход: {X[i]}, Предсказание: {prediction}, Ожидалось: {y[i]}")
 