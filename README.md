# HandWrittenDigitsClassification
Linear models for Handwritten Digits Classification

Implemented Binary and multiclass Logistic Regression on MNIST dataset. 

Training has been done using Stochastic Gradient descent, Mini batch gradient descent and gradient descent. 

code/LogisticRegression.py has implementation for Logisitc Regression.

code/LRM.py has implementation for multi-class Logisitc Regression.

Binary logisitc regression(Sigmoid) is equivalent to Multiclass logistic regression with k=2(Softmax). If the learning rate of sigmoid is
set as two times the learning rate of softmax, then w1-w2(Softmax weights) = w(Sigmoid weights) holds for all training steps.


