# HandWritten Digits Classification - MNIST dataset
Implemented Binary and multiclass Logistic Regression on MNIST dataset for 16x16 grayscale images. Training has been done using Stochastic Gradient descent, Mini batch gradient descent and gradient descent. 

## Steps to run the code
code/LogisticRegression.py has implementation for Logisitc Regression.

code/LRM.py has implementation for multi-class Logisitc Regression.

Run code/main.py for training, hyper-parameter tuning. Hyper-parameter tuning is done to determine the best learning rate, the best max number of iterations for training. With the best parameters obtained, the model is re-trained and run on the test data.

## Relation between Binary and Multiclass(k=2) logistic regression

Binary logisitc regression(Sigmoid) is equivalent to Multiclass logistic regression with k=2(Softmax). If the learning rate of sigmoid is
set as two times the learning rate of softmax, then w1-w2(Softmax weights) = w(Sigmoid weights) holds for all training steps.


