import math
from graphviz import Digraph


class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        
        self.label = label
        self.data = data
        self.grad = 0.0
        
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value({self.label}, data={self.data}, grad={self.grad})"
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __add__(self, other):
        
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # print(f'    Calling _backward() (+) on {self}')
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # print(f'    Calling _backward() (*) on {self}')
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        
        assert isinstance(other, (int, float)), "Called __pow__ with non-int/float argument!"
        
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            # power rule combined with chain rule
            self.grad += other * self.data**(other - 1) * out.grad
            
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return self * other**-1
     
    def tanh(self):

        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            # print(f'    Calling _backward() on {self}')
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad # d/dx(e^x) = e^x
        
        out._backward = _backward
        
        return out
    
    def backward(self):
    
        def topological_sort(v):

            ordering = []
            visited = set()

            def grow_topological(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        grow_topological(child)
                    ordering.append(v)    

            grow_topological(v)

            return ordering
    
        self.grad = 1.0

        topo = topological_sort(self)
        
        for v in reversed(topo):
            # print(f'Calling _backward() on {v}')
            v._backward()


def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges


def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{%s|data: %.4f|grad:%.4f}" % (n.label, n.data, n.grad), 
             shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

a = Value(2, label='a')
b = Value(-3.0, label='b')
c = Value(10, label='c')
e = a*b; e.label='e'
d = e + c; d.label='d'
f = Value(-2, label='f')
L = d*f; L.label='L'
draw_dot(L)