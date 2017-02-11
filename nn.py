import numpy as np
# import matplotlib.pylab as plt


def initialize_network(input_size, output_size, hidden_sizes):
    Ws = []
    cols = input_size
    for rows in hidden_sizes:
        Ws.append(initialize_layer(rows, cols + 1))
        cols = rows
    Ws.append(initialize_layer(output_size, cols + 1))

    return Ws


def initialize_layer(rows, cols):
    return (np.random.random([rows, cols]) - 0.5)


def add_bias(x):
    bias = np.ones((1, x.shape[1]))
    return np.append(bias, x, axis=0)


def forward_layer(W, x):
    return activate(W @ add_bias(x))


def forward_all(Ws, x):
    y = [x]

    for W in Ws:
        x = forward_layer(W, x)
        y.append(x)

    return y


def forward(x, Ws):
    return forward_all(Ws, x)[-1]


def activate(x):
    return 1/(1 + np.exp(-x))


def gradient(x):
    y = activate(x).transpose()
    return y*(1 - y)


def backpropagate_layer(x, y, W, learning_rate, delta):
    x_bias = add_bias(x)
    z = W@x_bias
    
    dZ = delta*gradient(z)
    dX = dZ@W
    dW = dZ.transpose()@x_bias.transpose()

    W = W + learning_rate*dW
    return dX[:,1:], W


def backpropagate(x, y, Ws, learning_rate):
    new_Ws = []

    xs = forward_all(Ws, x)
    delta = y - (xs[-1]).transpose()
    for W, x, y in zip(Ws[::-1], xs[-2::-1], xs[-1:0:-1]):
        delta, new_W = backpropagate_layer(x, y, W, learning_rate, delta)
        new_Ws.append(new_W)

    return new_Ws[::-1]


def calculate_error(x, y, Ws):
    z = forward(x, Ws).transpose()
    error = y - z
    squared_error = [e @ e.transpose() for e in error]
    return np.mean(squared_error)


def main():
    # np.random.seed(0)

    learning_rate = 0.1
    x = np.array([[1, 2, 3], [4, 5, 6]]).transpose()
    y = np.array([[1, 0, 1], [1, 0, 0.3]])
    hidden_sizes = [4, 15, 5]

    Ws = initialize_network(len(x), y.shape[1], hidden_sizes)
    errors = []

    for _ in range(20000):
        Ws = backpropagate(x, y, Ws, learning_rate)
        errors.append(calculate_error(x, y, Ws))
        if _ % 100 == 0:
            learning_rate = learning_rate * 0.99

    z = forward(x, Ws).transpose()
    print('z:', z)
    print('error:', y - z)

    # plt.plot(errors)
    # plt.show()

if __name__ == '__main__':
    main()
