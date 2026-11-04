# autograd-engine

A scalar reverse-mode automatic differentiation engine in about 150 lines of Python, with a tiny
neural-network library on top. Each `Value` holds one float, records the operation that produced
it, and knows its local derivative; `backward()` walks the graph in reverse topological order and
accumulates gradients with the chain rule.

Supported ops: `+ - * / **` (with Python numbers on either side), unary `-`, `exp`, `log`, `tanh`, `relu`.

## Usage

```python
from value import Value

x = Value(0.5)
y = (x * x + x).tanh()
y.backward()
print(x.grad)   # (1 - tanh(0.75)^2) * (2*0.5 + 1) = 1.1932
```

Train a 2-8-1 MLP on XOR (`mlp.py`, `mlp_test.py`):

```
python mlp_test.py   # loss falls from ~4 to ~0.03 in 100 epochs
```

## Tests

`test_gradients.py` checks every op against central finite differences, checks that a node used
twice receives both gradient contributions (`y = x*x + x`), and pins `relu'(0) = 0`.
From a notebook cell in this folder:

```python
import sys
!{sys.executable} -m pytest test_gradients.py -q
```

## Known gaps

- Scalar only: one Python object per number, so a 1024x1024 matmul would be about a million graph
  nodes and Python-level loops. Real frameworks put whole tensors in one node and run vectorized
  kernels; their backward must then also sum gradients over broadcast dimensions.
- `backward()` builds the topological order recursively, so graphs deeper than about 1000 nodes
  hit Python's recursion limit.
- Gradients are not reset by `backward()`; call `zero_grad()` before each step.
