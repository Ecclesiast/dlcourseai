# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Задание 3.1 - Сверточные нейронные сети (Convolutional Neural Networks)
# 
# Это последнее задание на numpy, вы до него дожили! Остался последний марш-бросок, дальше только PyTorch.
# 
# В этом задании вы реализуете свою собственную сверточную нейронную сеть.

# %%
import numpy as np
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# %%
from dataset import load_svhn, random_split_train_val
from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient
from layers import FullyConnectedLayer, ReLULayer, ConvolutionalLayer, MaxPoolingLayer, Flattener
from model import ConvNet
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from metrics import multiclass_accuracy

# %% [markdown]
# # Загружаем данные
# 
# На этот раз мы не будем их преобразовывать в один вектор, а оставим размерности (num_samples, 32, 32, 3).

# %%
def prepare_for_neural_network(train_X, test_X):    
    train_X = train_X.astype(np.float) / 255.0
    test_X = test_X.astype(np.float) / 255.0
    
    # Subtract mean
    mean_image = np.mean(train_X, axis = 0)
    train_X -= mean_image
    test_X -= mean_image
    
    return train_X, test_X
    
train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)    
train_X, test_X = prepare_for_neural_network(train_X, test_X)
# Split train into train and val
train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)

# %% [markdown]
# # Реализуем новые слои!
# 
# Сначала основной новый слой - сверточный (Convolutional layer). 
# Для начала мы реализуем его для только одного канала, а потом для нескольких.
# 
# Сверточный слой выполняет операцию свертки (convolution) с весами для каждого канала, а потом складывает результаты. 
# Возможно, поможет пересмотреть Лекцию 6 или внимательно прочитать
# http://cs231n.github.io/convolutional-networks/
# 
# Один из подходов к реализации сверточного слоя основан на том, что для конкретного "пикселя" выхода применение сверточного слоя эквивалентно обычному полносвязному.  
# Рассмотрим один такой "пиксель":
# ![image.png](attachment:image.png)
# 
# Он получает на вход   
# регион входа I размера `(batch_size, filter_size, filter_size, input_channels)`,  
# применяет к нему веса W `(filter_size, filter_size, input_channels, output_channels`
# и выдает `(batch_size, output_channels)`. 
# 
# Если:  
# - вход преобразовать в I' `(batch_size, filter_size*filter_size*input_channels)`,  
# - веса в W' `(filter_size*filter_size*input_channels, output_channels)`,  
# то выход "пикселе" будет эквивалентен полносвязному слою со входом I' и весами W'.
# 
# Осталось выполнить его в цикле для каждого пикселя :)

# %%
# TODO: Implement ConvolutionaLayer that supports only 1 output and input channel

# Note: now you're working with images, so X is 4-dimensional tensor of
# (batch_size, height, width, channels)

X = np.array([
              [
               [[1.0], [2.0]],
               [[0.0], [-1.0]]
              ]
              ,
              [
               [[0.0], [1.0]],
               [[-2.0], [-1.0]]
              ]
             ])

# Batch of 2 images of dimensions 2x2 with a single channel
print("Shape of X:",X.shape)

layer = ConvolutionalLayer(in_channels=1, out_channels=1, filter_size=2, padding=0)
print("Shape of W", layer.W.value.shape)
layer.W.value = np.zeros_like(layer.W.value)
layer.W.value[0, 0, 0, 0] = 1.0
layer.B.value = np.ones_like(layer.B.value)
result = layer.forward(X)

assert result.shape == (2, 1, 1, 1)
assert np.all(result == X[:, :1, :1, :1] +1), "result: %s, X: %s" % (result, X[:, :1, :1, :1])


# Now let's implement multiple output channels
layer = ConvolutionalLayer(in_channels=1, out_channels=2, filter_size=2, padding=0)
result = layer.forward(X)
assert result.shape == (2, 1, 1, 2)


# And now multple input channels!
X = np.array([
              [
               [[1.0, 0.0], [2.0, 1.0]],
               [[0.0, -1.0], [-1.0, -2.0]]
              ]
              ,
              [
               [[0.0, 1.0], [1.0, -1.0]],
               [[-2.0, 2.0], [-1.0, 0.0]]
              ]
             ])

print("Shape of X:", X.shape)
layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0)
result = layer.forward(X)
assert result.shape == (2, 1, 1, 2)

# %% [markdown]
# ## А теперь имплементируем обратный проход
# Возможно, это самое сложное место в курсе. Дальше будет лучше.
# 
# Раз выполнение сверточного слоя эквивалентно полносвязному слою для каждого "пикселя" выхода, то общий обратный проход эквивалентен обратному проходу каждого из таких "слоев".  
# Градиенты от каждого из этих "слоев" в каждом пикселе надо сложить в соответствующие пиксели градиента по входу, а градиенты весов сложить все вместе.

# %%
# First test - check the shape is right
layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0)
result = layer.forward(X)
d_input = layer.backward(np.ones_like(result))
assert d_input.shape == X.shape

# Actually test the backward pass
# As usual, you'll need to copy gradient check code from the previous assignment
layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0)
assert check_layer_gradient(layer, X)

layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0)
assert check_layer_param_gradient(layer, X, 'W')
layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0)
assert check_layer_param_gradient(layer, X, 'B')

# %% [markdown]
# Осталось реализовать дополнение нулями (padding).   
# Достаточно дополнить входной тензор нулями по сторонам. Не забудьте учесть это при обратном проходе!

# %%
layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=3, padding=1)
result = layer.forward(X)
# Note this kind of layer produces the same dimensions as input
assert result.shape == X.shape,"Result shape: %s - Expected shape %s" % (result.shape, X.shape)
d_input = layer.backward(np.ones_like(result))

assert d_input.shape == X.shape,"Result shape: %s - Expected shape %s" % (d_input.shape, X.shape)
layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=3, padding=1)
#assert check_layer_gradient(layer, X)
assert check_layer_param_gradient(layer, X, "W")

# %% [markdown]
# ## После следующего слоя вам уже будет все ни по чем - max pooling
# 
# Max Pooling - это слой, реализующий операцию максимума для каждого канала отдельно в окресности из `pool_size` "пикселей".
# 
# ![image](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png)
# 
# И напомним что такое stride.  
# Stride - это на сколько "пикселей" сдвигается окно на одном шаге.  
# Вот здесь, например, stride = 2
# 
# ![image.png](http://deeplearning.net/software/theano/_images/no_padding_strides.gif)
# 
# На практике, для max pooling значение stride часто равно pool size.

# %%
pool = MaxPoolingLayer(2, 2)
result = pool.forward(X)
#print(result)
assert result.shape == (2, 1, 1, 2)

assert check_layer_gradient(pool, X)

# %% [markdown]
# И на закуску - слой, преобразующий четырехмерные тензоры в двумерные.
# 
# Этот слой понадобится нам, чтобы в конце сети перейти от сверточных слоев к полносвязным.

# %%
flattener = Flattener()
result = flattener.forward(X)
assert result.shape == (2,8)

assert check_layer_gradient(flattener, X)

# %% [markdown]
# # Теперь есть все кирпичики, создаем модель

# %%
# TODO: In model.py, implement missed functions function for ConvNet model

# No need to use L2 regularization
model = ConvNet(input_shape=(32,32,3), n_output_classes=10, conv1_channels=2, conv2_channels=2)
loss = model.compute_loss_and_gradients(train_X[:2], train_y[:2])
check_layer_param_gradient(model.Conv1_Layer, train_X[:2], "W")
# TODO Now implement backward pass and aggregate all of the params
#check_model_gradient(model, train_X[:2], train_y[:2])

# %% [markdown]
# # Оптимизатор и код для тренировки 
# Должен заработать с кодом из прошлого задания без изменений!

# %%
model = ConvNet(input_shape=(32,32,3), n_output_classes=10, conv1_channels=2, conv2_channels=2)
dataset = Dataset(train_X[:16], train_y[:16], val_X[:16], val_y[:16])
trainer = Trainer(model, dataset, SGD(), batch_size=16, learning_rate=1e-4)

loss_history, train_history, val_history = trainer.fit()


# %%
plt.plot(train_history)
plt.plot(val_history)

# %% [markdown]
# # Последнее упражнение
# В качестве последнего упражнения мы доведем точность на тренировочном наборе данных до 100% на небольшом наборе данных.
# Сверточные сети требуют большого количества вычислений и аккуратной эффективной реализации, поэтому настоящие модели мы будем тренировать уже на PyTorch в следующем задании.
# %% [markdown]
# ## Итак, оверфитим маленький набор данных
# 

# %%
data_size = 128
model = ConvNet(input_shape=(32,32,3), n_output_classes=10, conv1_channels=2, conv2_channels=2)
dataset = Dataset(train_X[:data_size], train_y[:data_size], val_X[:data_size], val_y[:data_size])
# TODO: Change any hyperparamers or optimizators to reach 1.0 training accuracy in 50 epochs or less
# Hint: If you have hard time finding the right parameters manually, try grid search or random search!
trainer = Trainer(model, dataset, MomentumSGD(), learning_rate=1e-5, num_epochs=50, batch_size=64)

loss_history, train_history, val_history = trainer.fit()


# %%
plt.plot(train_history)
plt.plot(val_history)

# %% [markdown]
# Дальнейшие упражнения - уже на PyTorch, открывайте следующий notebook!
# 
# ![image.png](attachment:image.png)

# %%


