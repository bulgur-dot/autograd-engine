from value import Value


def numgrad(f, xs, h=1e-6):
    """
    Compute the numerical gradient of a function f at points xs using finite differences.

    Parameters:
    - f: A function that takes a list of Value objects and returns a Value object.
    - xs: A list of Value objects at which to compute the gradient.
    - h: A small step size for finite differences (default is 1e-6).

    Returns:
    - A list of numerical gradients corresponding to each input in xs.
    """
    grads = []
    for i, x in enumerate(xs):
        # Store the original value
        original_value = x.data

        # Compute f(x + h)
        x.data = original_value + h
        f_plus_h = f(xs).data

        # Compute f(x - h)
        x.data = original_value - h
        f_minus_h = f(xs).data

        # Restore the original value
        x.data = original_value

        # Compute the numerical gradient using central difference
        grad = (f_plus_h - f_minus_h) / (2 * h)
        grads.append(grad)

    return grads

def test_numgrad1():
    # Every op at least once: add, radd, mul, rmul, pow, neg, sub, rsub,
    # truediv, rtruediv, exp, log, tanh, relu. log's argument stays positive,
    # no division by anything near 0.
    def f(xs):
        x, y, z = xs
        a = x ** 2 + 2 * y.log() / y.exp() - z.tanh()   # pow, rmul, log, truediv, exp, sub, tanh
        b = (x * z).relu() + 1 - (-y)                  # mul, relu, add, neg, sub
        c = 3 / (z + 2) + (5 - x) * 0.5                # rtruediv, rsub, mul
        return a * b + c

    xs = [Value(1.5), Value(0.8), Value(-0.6)]

    expected = numgrad(f, xs)      # finite differences first: numgrad only runs f, no backward

    out = f(xs)                     # fresh forward graph
    out.backward()                  # backward() returns None; the gradients land on the inputs
    for i, (x, g) in enumerate(zip(xs, expected)):
        assert abs(x.grad - g) <= 1e-4, f"input {i}: autograd {x.grad}, numerical {g}"


def test_numgrad2():
    # Define a function for testing:
    def f(xs):
        return xs[0] ** 2 + xs[0]

    a = Value(3.0)
    y = f([a])  # Forward pass
    y.backward()

    grad_a = a.data * 2 + 1

    assert abs(a.grad - grad_a) < 1e-4, f"Expected gradient {grad_a}, but got {a.grad}"

def test_numgrad3():

    a = Value(0.0)
    a.relu().backward()

    assert a.grad == 0.0, f"Expected gradient 0.0, but got {a.grad}"


if __name__ == "__main__":
    print("Test 1:", test_numgrad1())
    print("Test 2:", test_numgrad2())
    print("Test 3:", test_numgrad3())