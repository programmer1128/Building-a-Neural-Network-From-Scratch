Multilayer Perceptron (MLP) from Scratch in C++


A complete, dependency-free implementation of a Multilayer Perceptron (MLP) Neural Network written purely in C++. This project builds everything from the ground up, including custom matrix operations, forward and backward propagation algorithms, loss calculations, and binary model serialization.Currently, the network architecture is configured to tackle the classic MNIST digit recognition problem (784 inputs -> 10 outputs).

🚀 Key Features
1)Zero Dependencies: No external linear algebra or machine learning libraries were used.

2)Custom Linear Algebra: Matrix multiplication, addition, subtraction, and transposition implemented from scratch.

3)Modern Weight Initialization: Uses He Normal Initialization to prevent vanishing/exploding gradients when using ReLU.

4)Numerical Stability: Softmax implementation includes a max-subtraction trick to prevent floating-point overflow.

5)Binary Serialization: Custom .save() and .load() functionality to export and import trained model weights as binary files with a custom MLP header.

Mathematical Foundations & Implementation
This section outlines the mathematics driving the network and how they are translated into the custom C++ codebase.

1. Weight Initialization (He Initialization)To ensure the network learns effectively when using ReLU activations, weights are initialized using He Initialization. This draws weights from a normal distribution with a mean of $0$ and a standard deviation of

   $\sqrt{\frac{2}{\text{input\_size}}}$.

C++ Implementation (Layer.c++):

C++
```cpp
// Using values from probability distribution of mean = 0
// and standard deviation = sqrt(2/input_size)
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, std::sqrt(2.0/input_size));

for(int i = 0; i < output_size; i++)
{
     for(int k = 0; k < input_size; k++)
    {
        weights[i][k] = distribution(generator);
    }
}

```



2. Forward PropagationData flows from the input layer through hidden layers to generate a prediction. For any given layer $l$, the linear combination of inputs, weights, and biases is calculated as
                           $$Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}$$

C++ Implementation (Layer.c++):

```cpp
// Z = W * A
std::vector<std::vector<double>> result(weights.size(), std::vector<double>(input[0].size()));
result = multiply_matrix(weights, input);

// Z = Z + b
for(int i = 0; i < result.size(); i++)
{
     for(int k = 0; k < result[0].size(); k++)
    {
        result[i][k] += (double)biases[i][k];
    }
}

```

3. Activation FunctionsHidden Layers (ReLU): To introduce non-linearity, we use the Rectified Linear Unit.
                                     $$f(x) = \max(0, x)$$

C++ Implementation (functions.cpp):

```cpp
double relu(double x)
{
     return std::max(0.0, x);
}

```


4. Output Layer (Softmax): To convert raw scores into a probability distribution, we use Softmax. To maintain numerical stability (preventing floating-point overflow from large exponentials), we subtract the maximum value in the output vector before exponentiation:
                  $$\sigma(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

C++ Implementation (functions.cpp):

```cpp
// Find max element for numerical stability
double max_val = -std::numeric_limits<double>::infinity();
for (int i = 0; i < z.size(); i++)
{
     for (int k = 0; k < z[0].size(); k++)
     {
         if (z[i][k] > max_val) max_val = z[i][k];
     }
}

// Compute sum of exp(x - max_val) and normalize
double sum = 0.0;
for (int i = 0; i < z.size(); i++) {
    for (int k = 0; k < z[0].size(); k++) {
        output[i][k] = std::exp(z[i][k] - max_val);
        sum += output[i][k];
    }
}
// Normalization loop follows...

```


5. Loss Function (Categorical Cross-Entropy)To measure how far off the network's predictions ($\hat{y}$) are from the actual true labels ($y$), the network calculates Categorical Cross-Entropy loss:
                    $$L = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

```cpp
C++ Implementation (Network.c++):


double sample_loss = 0.0;
for (int j = 0; j < 10; ++j)
{
    // Avoid log(0) by adding a small epsilon
    double p = std::max(output[j][0], 1e-15);
    sample_loss += -target[j][0] * std::log(p);
}

```


6. Backpropagation & OptimizationThe network learns by calculating the gradient of the loss function and moving backward through the layers. For the output layer using Softmax and Cross-Entropy, the error gradient ($\delta$) simplifies to the difference between predictions and targets:
                      $$\delta^{(L)} = \hat{y} - y$$

```cpp
C++ Implementation (Layer.c++ - Output Layer):

std::vector<std::vector<double>> result = output;
delta = result;
         
for(int i = 0; i < row; i++)
{
    for(int k = 0; k < col; k++)
    {
        delta[i][k] -= target[i][k]; // prediction - target
    }
}

```


For hidden layers, the error is propagated backward using the derivative of the ReLU function ($f'(Z^{(l)})$) and the transposed weights of the next layer:

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(Z^{(l)})$$

C++ Implementation (Layer.c++ - Hidden Layers):
```cpp
// (W^(l+1))^T * delta^(l+1)
std::vector<std::vector<double>> next_layer_weight_transpose = transpose_matrix(next_weights);
std::vector<std::vector<double>> propagated_error = multiply_matrix(next_layer_weight_transpose, next_delta);

// Element-wise multiplication with ReLU derivative
std::vector<std::vector<double>> relu_deriv = relu_derivative_of_matrix(output);
for(int i = 0; i < row; i++)
{
    for(int k = 0; k < col; k++)
    {
        propagated_error[i][k] *= relu_deriv[i][k];
    }
}
delta = propagated_error;

```


Finally, weights are updated using Stochastic Gradient Descent (SGD):
$$W^{(l)} = W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$


C++ Implementation (Layer.c++):

```cpp
void Layer::update_weights(double learning_rate, const std::vector<std::vector<double>> &error) {
    int row = weights.size(); int col = weights[0].size();
    for(int i = 0; i < row; i++)
    {
        for(int k = 0; k < col; k++)
        {
            weights[i][k] -= learning_rate * (error[i][k]);
        }
    }
}

```


📂 Project Structure
Network.hpp / Network.c++: The core controller. Assembles the Layer objects, manages the training loop (epochs), computes Cross-Entropy loss, tracks accuracy, and handles saving/loading the final model.

Layer.hpp / Layer.c++: Represents a single layer in the network. Manages its own weight and bias matrices, performs forward passes, and calculates gradients during backpropagation.

matrices.hpp / matrices.c++: A custom 2D vector-based linear algebra library handling matrix multiplication, addition, subtraction, and transpositions.

functions.hpp / functions.cpp: Contains activation functions (ReLU, Softmax) and their derivatives.

sigma_function.c++: Contains standard Sigmoid functions (available as an alternative activation).



Model Serialization
The network can be saved to disk so you don't have to retrain it for every inference run. The save_model function writes the topology and exact floating-point weights/biases into a binary file.

C++
Network my_net({{784, 128}, {128, 10}});
my_net.train_network(X_train, Y_train, 10, 0.0001);
my_net.save_model("mnist_model.bin");

// Later, or in a different program:
Network loaded_net;
loaded_net.load_model("mnist_model.bin");

Current Assumptions
Input Dimensions: The train_network loop in Network.c++ currently hardcodes the input vectors to a size of 784 and targets to 10. This is perfect for MNIST, but needs parameterization to handle arbitrary datasets.

Batch Size: The network currently processes training data sequentially (Stochastic Gradient Descent with a batch size of 1).


🛠️ Usage & Compilation
This project uses CMake for its build system and requires OpenCV to be installed on your machine.

Prerequisites
A modern C++ compiler supporting C++20.

CMake (v3.10.0 or higher).

OpenCV (Ensure OpenCV_DIR is correctly set in the CMakeLists.txt or your system environment variables to point to your OpenCV build directory).

Build Instructions

1)Clone the repository and navigate into it:

```Bash
git clone <your-repository-url>
cd <your-repository-directory>
```
2)Create a build directory:

```Bash
mkdir build
cd build
```
3)Generate the build files using CMake:
(Note: If your OpenCV is installed in a custom location, ensure the path in CMakeLists.txt is correct for your system before running this).

```Bash
cmake ..
```
4)Compile the executable:

```Bash
cmake --build .
```
This will generate the Neural_Network_From_Scratch executable in your build directory.
