from value import Value
from graph import trace, draw_dot


value = Value(2.0) + Value(3.0) * Value(4.0) ** 2 - Value(5.0) * Value(6.0)
print(value) 
value.backward()
draw_dot(value, format='svg', rankdir='LR').render('graph', view=True)