import numpy as np
import random


class NeuralNetwork:
    def __init__(self, input_size: int, neurons_count: int):
        """Инициализация нейронной сети
        :param input_size: количество входов
        :param neurons_count: количество нейронов
        """
        self.weights = np.array([
            [random.uniform(0.001, 0.2) for _ in range(input_size)] 
            for _ in range(neurons_count)
        ])

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Линейная функция активации"""
        return x

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Предсказание выхода сети
        :param inputs: входные данные
        :return: выходные данные
        """
        weighted_sum = np.dot(self.weights, inputs)
        return self._activation_function(weighted_sum)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000):
        """Обучение сети
        :param X: входные данные (матрица)
        :param y: целевые значения
        :param epochs: количество эпох обучения
        """
        for _ in range(epochs):
            for k in range(len(X)):
                output = self.predict(X[k])
                error = y[k] - output
                
                for i in range(self.weights.shape[0]):
                    numerator = error[i]
                    denominator = np.sum(self.weights[i] * X[k])
                    if denominator != 0:
                        self.weights[i] += numerator / denominator
