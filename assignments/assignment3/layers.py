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
    softmax = (prediction_exp.T / np.sum(prediction_exp, axis=1)).T

    loss = np.sum(-np.log(softmax[range(target_index.shape[0]), target_index])) / target_index.shape[0]

    softmax[range(target_index.shape[0]), target_index] -= 1
    return loss, softmax


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
        self.W = Param(np.random.randn(n_input,n_output)*np.sqrt(2/(n_output+n_input)))
        self.B = Param(np.random.randn(1,n_output)*np.sqrt(2/(n_output+1)))
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

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, step=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)*np.sqrt(2/(2*filter_size + in_channels + out_channels))
        )

        self.B = Param(np.random.randn(out_channels)*np.sqrt(2/out_channels))

        self.padding = padding
        self.step = step


    def forward(self, X):
        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        self.X = np.pad(X, pad_width=npad, mode='constant', constant_values=0)
        #self.X = X.copy()
        batch_size, height, width, channels = X.shape
        
        out_height = int((height + 2*self.padding - self.filter_size) / self.step) + 1
        out_width = int((width + 2*self.padding - self.filter_size) / self.step) + 1

        self.result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                current_step_h = self.step*y
                current_step_w = self.step*x
                self.result[:, y, x, :] = self.X[:, current_step_h:self.filter_size + current_step_h, current_step_w:self.filter_size + current_step_w, :].reshape(batch_size, -1).dot(self.W.value.reshape(-1, self.out_channels)) + self.B.value
        
        return self.result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        #print(self.X.shape)
        d_input = np.zeros(self.X.shape)
        for y in range(out_height):
            for x in range(out_width):
                current_step_h = self.step*y
                current_step_w = self.step*x
                # print("Input")
                
                # print(d_input[:, current_step_h:self.filter_size + current_step_h, current_step_w:self.filter_size + current_step_w, :].shape)

                # print("Weights")

                # print(self.W.value.shape)

                d_input[:, current_step_h:self.filter_size + current_step_h, current_step_w:self.filter_size + current_step_w, :] += \
                    d_out[:, y, x, :].reshape(batch_size, -1).dot(self.W.value.reshape(-1, out_channels).T) \
                    .reshape(d_input[:, current_step_h:self.filter_size + current_step_h, current_step_w:self.filter_size + current_step_w, :].shape)

                self.W.grad += self.X[:, current_step_h:self.filter_size + current_step_h, current_step_w:self.filter_size + current_step_w, :] \
                    .reshape(batch_size, -1).T.dot(d_out[:, y, x, :].reshape(batch_size, -1)) \
                    .reshape(self.W.grad.shape)
                self.B.grad += d_out[:, y, x, :].reshape(-1, out_channels).T.dot(np.ones(batch_size*1*1))
        #print(d_input)
        #print(self.W.grad)
        return d_input[:, self.padding:height - self.padding, self.padding:width - self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        #self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()

        self.out_height = int((height - self.pool_size) / self.stride) + 1
        self.out_width = int((width - self.pool_size) / self.stride) + 1

        output = np.zeros((batch_size, self.out_height, self.out_width, channels))
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        for y in range(self.out_height):
            for x in range(self.out_width):
                current_step_h = self.stride*y
                current_step_w = self.stride*x
                values = np.max(self.X[:, current_step_h:self.pool_size + current_step_h, current_step_w:self.pool_size + current_step_w, :].reshape(batch_size, -1, channels), axis=1)

                output[:, y, x, :] = values
        return output

    # def backward(self, d_out):
    #     # TODO: Implement maxpool backward pass
    #     batch_size, height, width, channels = self.X.shape

    #     d_input = np.zeros(self.X.shape)
    #     for bs in range(batch_size):
    #         for y in range(self.out_height):
    #             for x in range(self.out_width):
    #                 for c in range(channels):
    #                     current_step_h = self.stride*y
    #                     current_step_w = self.stride*x

    #                     source_array = self.X[bs, current_step_h:self.pool_size + current_step_h, current_step_w:self.pool_size + current_step_w, c]
    #                     # Create the mask from a_prev_slice (≈1 line)
    #                     mask = self.create_mask_from_window(source_array)
    #                     mask_mul = 0
    #                     if np.sum(mask) > 1:
    #                         mask_mul = np.multiply(mask, d_out[bs, y, x, c]) / 2
    #                     else:
    #                         mask_mul = np.multiply(mask, d_out[bs, y, x, c])
    #                     d_input[bs, current_step_h:self.pool_size + current_step_h, current_step_w:self.pool_size + current_step_w, c] += mask_mul

    #     return d_input

    def create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        
        mask = x == np.max(x)
        
        return mask
    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape

        d_input = np.zeros(self.X.shape)

        for y in range(self.out_height):
            for x in range(self.out_width):
                current_step_h = self.stride*y
                current_step_w = self.stride*x

                source_array = self.X[:, current_step_h:self.pool_size + current_step_h, current_step_w:self.pool_size + current_step_w, :].reshape(batch_size, -1, channels)
                idxs = np.argmax(source_array, axis=1)

                idxs_exp = np.expand_dims(idxs, axis=1)

                array = d_input[:, current_step_h:self.pool_size + current_step_h, current_step_w:self.pool_size + current_step_w, :].copy()
                d_input_shape = array.shape
                np.put_along_axis(array.reshape(batch_size, -1, channels), idxs_exp, d_out[:, y, x, :].reshape(batch_size, -1, channels), axis=1)


                d_input[:, current_step_h:self.pool_size + current_step_h, current_step_w:self.pool_size + current_step_w, :] = array.reshape(d_input_shape)

        
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
