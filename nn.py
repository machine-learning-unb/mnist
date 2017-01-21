import numpy as np


def initialize_network(input_size, output_size, hidden_sizes):
    Ws = []
    cols = input_size
    for rows in hidden_sizes:
        Ws.append(initialize_layer(rows, cols + 1))
        cols = rows
    Ws.append(initialize_layer(output_size, cols + 1))

    return Ws


def initialize_layer(rows, cols):
    return np.random.random([rows, cols])


def forward_layer(W, x):
    return activate(W @ np.append(1, x))


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
    return activate(x)*(1 - activate(x))


def backpropagate_layer(x, y, W, learning_rate, delta):
    print('Delta:', delta.shape)
    print('Gradient:', gradient(y).shape)
    print('X:', np.append(1, x).shape)

    dX = delta@gradient(y)*W
    dW = delta@gradient(y)*np.append(1, x)

    print('dX', dX.shape)
    print('dW', dW.shape)

    W = W - learning_rate*dW
    return dX, W


def backpropagate(x, y, Ws, learning_rate):
    new_Ws = []

    xs = forward_all(Ws, x)
    delta = y - xs[-1]

    for W, x, y in zip(Ws[::-1], xs[-2::-1], xs[-1:1:-1]):
        delta, W = backpropagate_layer(x, y, W, learning_rate, delta)
        new_Ws.append(W)

    return new_Ws[::-1]


def main():
    np.random.seed(0)

    learning_rate = 0.1
    x = np.array([1, 2, 3]).transpose()
    y = np.array([1, 0, 1]).transpose()
    hidden_sizes = [4, 7]

    Ws = initialize_network(len(x), len(y), hidden_sizes)

    for w in Ws:
        print(w.shape)

    z = forward(x, Ws)
    Ws = backpropagate(x, y, Ws, learning_rate)

    print(Ws)

if __name__ == '__main__':
    main()
