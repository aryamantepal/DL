import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/train.csv")

df.head()

df = np.array(df)
m, n = df.shape


np.random.shuffle(df)
df_dev = df[0:1000].T
Y_dev = df_dev[0]
X_dev = df_dev[1:n]

df_train = df[100:m].T
Y_train = df_train[0]
X_train = df_train[1:n]

# normalizing
X_train = X_train / 255.
X_dev = X_dev / 255.

def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(1. / 784)
    W2 = np.random.randn(10, 10) * np.sqrt(1. / 10)
    b1 = np.random.rand(10, 1)
    # W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

# def SoftMax(Z2):
#     return np.exp(Z2) / np.sum(np.exp(Z2))
def SoftMax(Z2):
    expZ = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))  # numerical stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = SoftMax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    dZ2 = A2 - one_hot_Y                             # (num_classes, m)
    dW2 = (1 / m) * dZ2.dot(A1.T)                    # (num_classes, hidden_units)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)             # (hidden_units, m)
    dW1 = (1 / m) * dZ1.dot(X.T)                     # (hidden_units, input_dim)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions==Y) / Y.size

def get_predictions(A2):
    return np.argmax(A2, 0)

def gradient_descent(X, Y, epochs, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # if i % 10 == 0:
            # print("iteration:", i)
        print("accuracy:", i, get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2
    
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)