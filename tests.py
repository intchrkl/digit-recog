import numpy as np
from layers import Linear, ReLU
from loss import SoftmaxCrossEntropy

def LinearTest():
    layer = Linear(4, 3)
    assert(layer.W.shape == (3, 4))
    x = np.random.randn(4, 1) # input w/ 4 features
    out = layer.forward(x)
    assert(out.shape == (3, 1))
    grad_out = np.ones((3, 1))
    dx = layer.backward(grad_out)
    assert(dx.shape == (4, 1))
    layer.step()

def ReluTest():
    relu = ReLU()
    x = np.array([[1.0], [-0.5], [2.0], [0.0]])
    out = relu.forward(x)
    assert(out[0] == [1.0])
    assert(out[1] == [0.0])
    assert(out[2] == [2.0])
    assert(out[3] == [0.0])
    grad_out = np.ones_like(x)
    dx = relu.backward(grad_out)
    assert(dx[0] == [1.0])
    assert(dx[1] == [0.0])
    assert(dx[2] == [1.0])
    assert(dx[3] == [0.0])
    
def SoftmaxCrossEntropyTest():
    sce = SoftmaxCrossEntropy()
    logits = np.array([2.0, 1.0, 0.1])
    label = 0
    probs, loss = sce.forward(logits, label)
    assert(sum(probs) == 1.0)

LinearTest()
ReluTest()
SoftmaxCrossEntropyTest()