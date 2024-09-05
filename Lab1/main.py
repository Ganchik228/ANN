import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = sum(inputs[i] * self.weights[i] for i in range(len(inputs))) + self.bias
        return self.activate(weighted_sum)

if __name__ == '__main__':
    req = [1, 1, 0, 1]
    neuroReq = []
    while req!=neuroReq:
        neuroReq.clear()
        neuron = Neuron(2)
        test = [[0, 0], [0, 1], [1, 0], [1, 1]]

        print("Веса:", neuron.weights)
        print("Смещение:", neuron.bias)
        print("\nРезультаты:")
        for inputs in test:
            predict = neuron.predict(inputs)
            neuroReq.append(predict)
            print(f"Входы: {inputs}, Выход: {predict}")
            