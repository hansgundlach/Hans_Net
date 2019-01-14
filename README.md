# Hans_Net
HansNet is an implementation of a Basic Neural Network in Java.
HansNet is designed for basic multi-dimensional regression.
Currently HansNet does not implement any regularization functions.
Weights are adjusted using stochastic gradient descent.
The project is divided into two classes. The first class is Neural_Net which contains
methods necessary to train a neural network ie implementation of stochastic gradient descent. The second class is Mat which deals with all the basic matrix
operations necessary in a neural network ie transpose, multiplication, addition, etc.
For example:
```Java
double inputData[][] = new double[][] {
 {
  .1,
  .1
 }, {
  .2,
  .2
 }, {
  10,
  10
 }, {
  20,
  20
 }
};
double outputData[][] = new double[][] {
 {
  10
 }, {
  10
 }, {
  99
 }, {
  105
 }
};


Neural_Net firstnet = new Neural_Net();
firstnet.train(inputData,outputData);
firstnet.prediction(inputData);
```
