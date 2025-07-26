A Multilayer Perceptron Model Implemented From Absolute Scratch

Project Structure 

main.cpp - Handles The input output of the image files 
Layer.cpp - implements the forward propagation of input and the backpropagation  for updating the weights during training
Network.cpp - Manages all the layers together as a Network 
matrices.cpp - handles all the matrix operations like multiplication, addition, subtraction, transpose
functions.cpp - Contains all the functions for the mathematical operations along with their derivatives implemented for backpropagation

Inputs are passed as a vector. Matrix manipulation is the key here. To replicate biological neurons and synapses, we need an Activation Function. I chose to use the Sigmoid Function 
as I was familiar with it.

sigmoid(x) = 1/1+e^-x

Basically the output from each neuron happens to be the sigmoid of the weighted sum and biases.For simplicity I have considered only weights for now.

Then, I had to design an error function. This points out the networks mistakes and it's accuracy in giving the desired output. Common intuition says if we can reduce the error we get 
a better accuracy, which implies our network is learning to get better.. Here i used the concept of gradient in calculus for minima of a function to reduce the error which in AI terms 
is backpropagation.

Each node in one layer is connected to all the other nodes in the next layer. So one node will have weights= total number of nodes in the next layer

How does the input propagate?

The output of one layer is input for the next . As there is a collection of nodes and each revieves a input which is the weighted sum of the previous layer inputs and respective weights
the output can be described as the multiplication of the weight matrix and the input matrix to a layer

