# MNIST
Automatic handwritten digit recognizer using Artificial Neural Networks.

This is our first Machine Learning @ UnB programming exercise, so we decided to take a shot in the most famous dataset in Computer Vision: [MNIST handwritten digits](https://www.kaggle.com/c/digit-recognizer) made available by [Kaggle](https://www.kaggle.com). The idea is simple: our system must automatically identify which digit each 28x28 image corresponds to.

We decided to implement an [Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) and train it with the [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation). No hard decisions on the neural network architecture were made yet, such as number of neurons or hyperparameters values.

## History

### 03-Feb-2017

In our second meeting, we could finally make a neural network learn from *dummy* data. We implemented forward pass, backpropagation and error visualization. However the neural network isn't overfitting the dummy data (it was expected to do so since we feed it with the same values over and over again) and we don't know how to explain it.

As the next steps, we should review the backpropagation algorithm, refactor the code to be more maintainable and readable, as well as test it with MNIST data.

### 20-Jan-2017

In this first meeting, we could take a grasp on how a neural network processes data and how it can be trained to perform better in the long run. Concepts such as layers, input, output, perceptron, weights, feedforward, backpropagation and learning rate were discussed.

We started implementing our neural network in [Python](https://www.python.org) using the procedural paradigm. [NumPy](http://www.numpy.org) was our numerical package of choice for simpler Linear Algebra computation. We could implement the feedforward for N hidden layers but stuck in implementing backpropagation. It seems to work for the last layer but cannot keep doing for more hidden layers due to matrix dimensions mismatch.

## References:

- [Kaggle - Digit recognizer](https://www.kaggle.com/c/digit-recognizer)
- [Wikipedia - Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Wikipedia - Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [Matt Mazur - A step by step backpropagation example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- [Neural Networks and Deep Learning - Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Neural Networks and Deep Learning - How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)