#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys
import math

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        #Converting labels to one-hot vectors
        y = np.empty([n_samples, self.k])
        for i in range(n_samples):
            y[i] = np.eye(1, self.k, int(labels[i]))

        self.W = np.zeros([self.k, n_features])
        for n in range(self.max_iter):
            #To chose batch_size number of random samples from the input
            random_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_gradient = np.zeros([self.k, n_features])
            for idx in random_indices:
                batch_gradient = np.add(batch_gradient, self._gradient(X[idx], y[idx]))
            batch_gradient = batch_gradient / (batch_size)
            self.W = self.W - (self.learning_rate) * (batch_gradient)
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        #This function calculates gradient with respect to weight vector self.W and returns a matrix of
        #shape self.k x n_features i.e. the gradient with respect to each weight vector
        n_features = _x.shape[0]
        gradient_result = np.empty([self.k, n_features])
        softmax_x = self.softmax(_x)
        for i in range(self.k):
            gradient_result[i] = (softmax_x[i] - _y[i]) * (_x)
        return gradient_result
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        #Returns a list of size self.k, where each value corresponds to the softmax probability of each class
        softmax_components = []
        sum_softmax_components = 0
        for j in range(self.k):
            softmax_component = math.exp(np.matmul(self.W[j].T, x))
            sum_softmax_components = sum_softmax_components + softmax_component
            softmax_components.append(softmax_component)
        softmax_components_new = [x / (sum_softmax_components) for x in softmax_components]
        return softmax_components_new
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


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = np.empty(n_samples)
        for i in range(n_samples):
            softmax_prediction = self.softmax(X[i])
            #Softmax function returns a list of probablities for all classes, pick the index with the highest value
            preds[i] = softmax_prediction.index(max(softmax_prediction))
        return preds
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        n_samples = X.shape[0]
        num_correct_samples = np.sum(self.predict(X) == labels)
        return num_correct_samples / n_samples
		### END YOUR CODE

