from cmath import e
from typing import Tuple, Literal

class Value:

    def __init__(self,
                 data: float,
                 sources: Tuple["Value", "Value"] = tuple(),
                 op: Literal["x", "+", "="] = "=",
                 gradient: float = 0.0,
                 ):
        self.data = data
        self.sources = sources
        self.op = op
        self.gradient = gradient
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def _local_gradient(self):
        if not self.sources:
            return 0, 0
        left_source, right_source = self.sources
        if self.op == "*":
            return right_source.data, left_source.data
        elif self.op == "+":
            return 1, 1
        elif self.op == "-":
            return 1, -1
        return 0, 0
                
    def __add__(self, val: "Value") -> "Value":
        res = Value(self.data + val.data, tuple([self, val]), "+")    
        return val

    def __subtract__(self, val: "Value") -> "Value":
        res = Value(self.data - val.data, tuple([self, val]), "-")
        return val

    def __mul__(self, val: "Value") -> "Value":
        res = Value(self.data * val.data, tuple([self, val]), "x")
        return res
    
    def chain_rule(self):
        self.gradient = 1.0
        if not self.sources:
            return

        left, right = self.sources
        left_local_grad, right_local_grad = self._local_gradient()

        left.gradient += self.gradient * left_local_grad
        right.gradient += self.gradient * right_local_grad

        left.chain_rule()
        right.chain_rule()

    def tanh_gradient(self):
        tanh_val = (e(self.data) - e(-self.data)) / (e(self.data) + e(-self.data))
        return 1 - tanh_val**2

    def plot(self, filename="computation_graph", view=False):
        return ""


# Example Usage:
x = Value(5)
y = Value(10)
z = x + y
print(z)

w = z * x
print(w)
