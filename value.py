import numpy as np


class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = .0
        self._children = children
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, children=(self,), op='**')
        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-1 * other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def exp(self):
        exp_res = np.exp(self.data)
        out = Value(exp_res, children=(self,), op='exp')
        def _backward():
            self.grad += exp_res * out.grad
        out._backward = _backward
        return out

    def log(self):
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out = Value(np.log(self.data), children=(self,), op='log')
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out = Value(t, children=(self,), op='tanh')
        out._backward = _backward
        return out

    def relu(self):
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out = Value(self.data if self.data > 0 else 0, children=(self,), op='ReLU')
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
