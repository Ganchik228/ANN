import numpy as np

class Neuron:

    def __init__(self, w: np.ndarray[float], b: float):
        if w.size>0:
            self.weight = w
        if b:
            self.bias = b

    def _threshold_function(self, x: np.ndarray[float]) -> int:
        return 1 if x >= 0 else 0

    def predict(self, x: np.ndarray[int]) -> int:
        u = 0
        u = sum(x[i] * self.weight[i] for i in range(len(x)))
        return self._threshold_function(u + self.bias)


if __name__ == "__main__":
    neuro = Neuron(w=np.array([-0.165, 0.766]), b=0.062)
    print(neuro.predict(x=np.array([0, 0])))
    print(neuro.predict(x=np.array([0, 1])))
    print(neuro.predict(x=np.array([1, 0])))
    print(neuro.predict(x=np.array([1, 1])))
    