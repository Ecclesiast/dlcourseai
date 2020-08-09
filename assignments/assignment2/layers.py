import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value
    
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    prediction_exp = np.exp(predictions)
    softmax = prediction_exp / np.sum(prediction_exp, axis=0)

    loss = -np.log(softmax[target_index])

    dprediction = softmax
    dprediction[target_index] -= 1

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.result = X.copy()

        self.result[self.result < 0] = 0

        return self.result

    def backward(self, d_out):

        return (self.result > 0) * d_out

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        result = self.X.dot(self.W.value) + self.B.value

        return result

    def backward(self, d_out):

        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.ones((1, d_out.shape[0])).dot(d_out)

        d_input = d_out.dot(self.W.value.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }