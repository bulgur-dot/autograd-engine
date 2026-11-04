from mlp import MLP, Value
import numpy as np


xs = [[0,0],[0,1],[1,0],[1,1]]
ys = [-1, 1, 1, -1]

mlp = MLP(2, [8, 1])

np.random.seed(42)

for epoch in range(100):
    loss = Value(0.0)
    for x, y in zip(xs, ys):
        # Forward pass
        out = mlp(x)
        loss += (out - y) ** 2

    # Backward pass
    mlp.zero_grad()
    loss.backward()

        # Update parameters
    for p in mlp.parameters():
        p.data -= 0.1 * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")