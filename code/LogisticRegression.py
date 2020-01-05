import numpy as np
import sys
import math

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        #Initializing the weights to zeros, as logistic regression has a convex optimization function and hence the
        #solution will converge.
        weights_initial = np.zeros(n_features)
        self.assign_weights(weights_initial)
        for n in range(self.max_iter):
            gradient_feature = np.zeros(n_features)
            #Iterate through all the samples to find the gradient
            for i in range(n_samples):
                gradient_feature = np.add(gradient_feature, self._gradient(X[i],y[i]))
            gradient_feature = (gradient_feature) / (n_samples)
            self.W = self.W - (self.learning_rate) * (gradient_feature)
		### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        #Update the weights after running gradient descent over a batch of random samples
        n_samples, n_features = X.shape
        weights_initial = np.zeros(n_features)
        self.assign_weights(weights_initial)
        for n in range(self.max_iter):
            batch_gradient = np.zeros(n_features)
            #To select batch_size number of random samples from the input
            random_indices = np.random.choice(n_samples, batch_size, replace=False)
            for idx in random_indices:
                batch_gradient = np.add(batch_gradient, self._gradient(X[idx], y[idx]))
            batch_gradient = batch_gradient / (batch_size)
            self.W = self.W - (self.learning_rate) * (batch_gradient)
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        #For stochastic gradient descent, pick a random training sample in each iteration 
        #and update the weights using the gradient for that sample
        n_samples, n_features = X.shape
        weights_initial = np.zeros(n_features)
        self.assign_weights(weights_initial)
        for n in range(self.max_iter):
            random_index = np.random.choice(n_samples, replace=False)
            sgd_gradient = self._gradient(X[random_index], y[random_index])
            self.W = self.W - (sgd_gradient) * (self.learning_rate)
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        denominator = 1 + math.exp(_y * (np.matmul(self.W.T, _x)))
        gradient = ((-1) * (_y) / (denominator)) * _x
        return gradient
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        n_samples = X.shape[0]
        preds_proba = np.zeros([n_samples, 2])
        for i in range(n_samples):
            preds_proba[i][0] = self.sigmoid(np.matmul(self.W.T, X[i]))
            preds_proba[i][1] = 1 - preds_proba[i][0]
        return preds_proba
		### END YOUR CODE

    #Additional function written to compute the sigmoid during prediction of probabilities    
    def sigmoid(self, _x):
        '''
        Calculates the sigmoid of the given value _x
        '''
        return 1 / (1 + math.exp(-_x))

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        #First predict the probability that each sample is +1
        n_samples = X.shape[0]
        predict_X = self.predict_proba(X)
        preds = np.zeros(n_samples)
        for i in range(n_samples):
            preds[i] = 1 if predict_X[i][0] > 0.5 else -1
        return preds
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        #To score the function, first calculate the predicted y labels
        y_predict = self.predict(X)
        score = np.mean(y_predict == y)
        return score
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

