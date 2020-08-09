import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )

from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient

class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.input_shape = input_shape
        self.n_output_classes = n_output_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels

        self.Conv1_Layer = ConvolutionalLayer(in_channels=input_shape[2], out_channels=self.conv1_channels, filter_size=3, padding=1)
        self.ReLU1_Layer = ReLULayer()
        self.MaxPool1_Layer = MaxPoolingLayer(4, 4)
        self.Conv2_Layer = ConvolutionalLayer(in_channels=self.conv1_channels, out_channels=self.conv2_channels, filter_size=3, padding=1)
        self.ReLU2_Layer = ReLULayer()
        self.MaxPool2_Layer = MaxPoolingLayer(4, 4)
        self.Flatten_Layer = Flattener()
        self.FC_Layer = FullyConnectedLayer(n_input=2*2*self.conv2_channels, n_output=n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        params["conv1_W"].grad = np.zeros(params["conv1_W"].grad.shape)
        params["conv1_B"].grad = np.zeros(params["conv1_B"].grad.shape)

        params["conv2_W"].grad = np.zeros(params["conv2_W"].grad.shape)
        params["conv2_B"].grad = np.zeros(params["conv2_B"].grad.shape)

        params["fc_W"].grad = np.zeros(params["fc_W"].grad.shape)
        params["fc_B"].grad = np.zeros(params["fc_B"].grad.shape)
        conv1 = self.Conv1_Layer.forward(X)
        relu1 = self.ReLU1_Layer.forward(conv1)
        maxpool1 = self.MaxPool1_Layer.forward(relu1)
        conv2 = self.Conv2_Layer.forward(maxpool1)
        relu2 = self.ReLU2_Layer.forward(conv2)
        maxpool2 = self.MaxPool2_Layer.forward(relu2)
        flatten = self.Flatten_Layer.forward(maxpool2)
        fc = self.FC_Layer.forward(flatten)
        loss, dpred = softmax_with_cross_entropy(fc, y)

        d_fc = self.FC_Layer.backward(dpred)
        d_flatten = self.Flatten_Layer.backward(d_fc)
        d_maxpool2 = self.MaxPool2_Layer.backward(d_flatten)
        d_relu2 = self.ReLU2_Layer.backward(d_maxpool2)
        d_conv2 = self.Conv2_Layer.backward(d_relu2)
        d_maxpool1 = self.MaxPool1_Layer.backward(d_conv2)
        d_relu1 = self.ReLU1_Layer.backward(d_maxpool1)
        d_conv1 = self.Conv1_Layer.backward(d_relu1)

        return loss
        
    def predict(self, X):
        conv1 = self.Conv1_Layer.forward(X)
        relu1 = self.ReLU1_Layer.forward(conv1)
        maxpool1 = self.MaxPool1_Layer.forward(relu1)
        conv2 = self.Conv2_Layer.forward(maxpool1)
        relu2 = self.ReLU2_Layer.forward(conv2)
        maxpool2 = self.MaxPool2_Layer.forward(relu2)
        flatten = self.Flatten_Layer.forward(maxpool2)
        fc = np.argmax(self.FC_Layer.forward(flatten), axis=1)

        return fc

    def params(self):
        result = {
            "conv1_W": self.Conv1_Layer.params()['W'], "conv1_B": self.Conv1_Layer.params()['B'], 
            "conv2_W": self.Conv2_Layer.params()['W'], "conv2_B": self.Conv2_Layer.params()['B'], 
            "fc_W": self.FC_Layer.params()['W'], "fc_B": self.FC_Layer.params()['B'],
        }

        return result
