import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    #take all the y indices for which class = 1
    y_class_1 = np.where(y == 1)[0]
    y_class_2 = np.where(y == -1)[0]
    class_1_samples = X[y_class_1, :]
    class_2_samples = X[y_class_2, :]
    plt.scatter(class_1_samples[:, 0], class_1_samples[:, 1], label = 'Class +1')
    plt.scatter(class_2_samples[:, 0], class_2_samples[:, 1], label = 'Class -1')
    plt.xlabel('Symmetry feature')
    plt.ylabel('Intensity feature')
    plt.title('Scatter plot of training features')
    plt.legend()
    plt.plot()
    #plt.savefig('train_features.png')
    #plt.show()
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ###YOUR CODE HERE
    y_class_1 = np.where(y == 1)[0]
    y_class_2 = np.where(y == -1)[0]
    class_1_samples = X[y_class_1, :]
    class_2_samples = X[y_class_2, :]
    plt.scatter(class_1_samples[:, 0], class_1_samples[:, 1], label = 'Class +1')
    plt.scatter(class_2_samples[:, 0], class_2_samples[:, 1], label = 'Class -1')
    x_values = [np.min(X[:, 0]), np.max(X[:, 1])]
    y_values = - (W[0] + np.dot(W[1], x_values)) / W[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Symmetry feature')
    plt.ylabel('Intensity feature')
    plt.legend()
    plt.title('Sigmoid model after training')
    #plt.savefig('train_result_sigmoid.png')
    #plt.show()
    ###END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    y_class_1 = np.where(y == 0)[0]
    y_class_2 = np.where(y == 1)[0]
    y_class_3 = np.where(y == 2)[0]
    class_1_samples = X[y_class_1, :]
    class_2_samples = X[y_class_2, :]
    class_3_samples = X[y_class_3, :]
    plt.scatter(class_1_samples[:, 0], class_1_samples[:, 1], label = 'Class 0')
    plt.scatter(class_2_samples[:, 0], class_2_samples[:, 1], label = 'Class 1')
    plt.scatter(class_3_samples[:, 0], class_3_samples[:, 1], label = 'Class 2')
    x_values = [np.min(X[:, 0]), np.max(X[:, 1])]
    for j in range(3):
        y_values = - (W[j][0] + np.dot(W[j][1], x_values)) / W[j][2]
        plt.plot(x_values, y_values, label='Decision Boundary')
        plt.ylim(-5,5)
    plt.xlabel('Symmetry feature')
    plt.ylabel('Intensity feature')
    plt.legend()
    plt.title('Softmax model after training')
    #plt.savefig('train_result_softmax.png')
    #plt.show()
    ### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)

   # ------------Logistic Regression Sigmoid Case------------

   ##### Check GD, SGD, BGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_GD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    #For Gradient Descent
    learning_rates = [0.1, 0.3, 0.4, 0.6, 0.7, 0.8]
    max_iterations = [10, 50, 100, 200, 500, 750, 1000]
    max_score_gd = 0
    best_learning_rate_gd = 0
    best_max_iterations_gd = 0
    best_logisticR_gd = None
    for learning_rate in learning_rates:
        for iterations in max_iterations:
            logisticR_classifier = logistic_regression(learning_rate=learning_rate, max_iter=iterations)
            logisticR_classifier.fit_GD(train_X, train_y)
            score_gd = logisticR_classifier.score(valid_X, valid_y)
            if score_gd > max_score_gd:
                max_score_gd = score_gd
                best_learning_rate_gd = learning_rate
                best_max_iterations_gd = iterations
                best_logisticR_gd = logisticR_classifier
    print("Gradient descent hyper-parameter tuning: learning_rate {}, max_iter {}, score {}".format(best_learning_rate_gd, best_max_iterations_gd, max_score_gd))
    print("Weights: {}".format(best_logisticR_gd.get_params()))

    #For Stochastic Gradient descent
    max_score_sgd = 0
    best_learning_rate_sgd = 0
    best_max_iterations_sgd = 0
    best_logisticR_sgd = None
    for learning_rate in learning_rates:
        for iterations in max_iterations:
            logisticR_classifier = logistic_regression(learning_rate=learning_rate, max_iter=iterations)
            logisticR_classifier.fit_SGD(train_X, train_y)
            score_sgd = logisticR_classifier.score(valid_X, valid_y)
            if score_sgd > max_score_sgd:
                max_score_sgd = score_sgd
                best_learning_rate_sgd = learning_rate
                best_max_iterations_sgd = iterations
                best_logisticR_sgd = logisticR_classifier
    print("Stochastic Gradient descent hyper-parameter tuning: learning_rate {}, max_iter {}, score {}".format(best_learning_rate_sgd, best_max_iterations_sgd, max_score_sgd))
    print("Weights: {}".format(best_logisticR_sgd.get_params()))

    #For batch gradient descent
    batch_sizes = [1, 10, 100, 200, 500, 1000, data_shape]
    max_score_bgd = 0
    best_learning_rate_bgd = 0
    best_max_iterations_bgd = 0
    best_batch_size_bgd = 0
    best_logisticR_bgd = None
    for learning_rate in learning_rates:
        for iterations in max_iterations:
            for batch_size in batch_sizes:
                logisticR_classifier = logistic_regression(learning_rate=learning_rate, max_iter=iterations)
                logisticR_classifier.fit_BGD(train_X, train_y, batch_size)
                score_bgd = logisticR_classifier.score(valid_X, valid_y)
                if score_bgd > max_score_bgd:
                    max_score_bgd = score_bgd
                    best_learning_rate_bgd = learning_rate
                    best_max_iterations_bgd = iterations
                    best_logisticR_bgd = logisticR_classifier
                    best_batch_size_bgd = batch_size
    print("Batch Gradient descent hyper-parameter tuning: learning_rate {}, max_iter {}, batch_size {}, score {}".format(best_learning_rate_bgd, best_max_iterations_bgd, best_batch_size_bgd, max_score_bgd))
    print("Weights: {}".format(best_logisticR_bgd.get_params()))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    ### YOUR CODE HERE
    #The best model obtained from hyper parameter training has been stored
    #Visualizing results for all the three models
    visualize_result(train_X[:, 1:3], train_y, best_logisticR_gd.get_params())
    visualize_result(train_X[:, 1:3], train_y, best_logisticR_sgd.get_params())
    visualize_result(train_X[:, 1:3], train_y, best_logisticR_bgd.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))

    test_X_all = prepare_X(raw_test_data)
    test_y_all, test_idx = prepare_y(test_labels)  

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]

    test_y[np.where(test_y==2)] = -1
    #Testing for best models on all three types of gradient descent
    print("Accuracy on test for best models: GD {}, SGD {}, BGD {}".format(best_logisticR_gd.score(test_X, test_y), best_logisticR_sgd.score(test_X, test_y), best_logisticR_bgd.score(test_X, test_y)))
    ### END YOUR CODE

    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    learning_rates_multi = [0.1, 0.3, 0.4, 0.6, 0.7, 0.8]
    max_iterations_multi = [100, 200, 500, 1000, 2000]
    batch_sizes_multi = [10, 100, 500, 1000, train_y.shape[0]]
    max_score_bgd_multi = 0
    best_learning_rate_bgd_multi = 0
    best_max_iterations_bgd_multi = 0
    best_batch_size_bgd_multi = 0
    best_logistic_multi_R = None
    for learning_rate in learning_rates_multi:
        for iterations in max_iterations_multi:
            for batch_size in batch_sizes_multi:
                logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=learning_rate, max_iter=iterations, k=3)
                logisticR_classifier_multiclass.fit_BGD(train_X, train_y, batch_size)
                score_bgd_multi = logisticR_classifier_multiclass.score(valid_X, valid_y)
                if score_bgd_multi > max_score_bgd_multi:
                    max_score_bgd_multi = score_bgd_multi
                    best_learning_rate_bgd_multi = learning_rate
                    best_max_iterations_bgd_multi = iterations
                    best_batch_size_bgd_multi = batch_size
                    best_logistic_multi_R = logisticR_classifier_multiclass
    print("Best learning rate for batch gradient descent after hyper-parameter tuning is {}".format(best_learning_rate_bgd_multi))
    print("Best max iterations for batch gradient descent after hyper-parameter tuning is {}".format(best_max_iterations_bgd_multi))
    print("Best score on batch gradient descent after hyper-parameter tuning is {}".format(max_score_bgd_multi))
    print("Best batch size after hyper-parameter tuning is {}".format(best_batch_size_bgd_multi))
    print("Weights of best model: {}".format(best_logistic_multi_R.get_params()))

    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())
    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_test_data)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all
    test_y = test_y_all
    print("Accuracy on multiclass using the best model is {}".format(best_logistic_multi_R.score(test_X, test_y)))
    print("Accuracy on multiclass using the best model for training set is {}".format(best_logistic_multi_R.score(train_X, train_y)))
    ### END YOUR CODE

    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 
    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000,  k= 2)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, train_y.shape[0])
    #Evaluate
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_test_data)
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx] 
    test_y[np.where(test_y==2)] = 0

    print("Softmax for 2 class convergence weights: {}".format(logisticR_classifier_multiclass.get_params()))
    print("Scores on training: {}, validation: {}, testing: {}".format(logisticR_classifier_multiclass.score(train_X, train_y), logisticR_classifier_multiclass.score(valid_X, valid_y), logisticR_classifier_multiclass.score(test_X, test_y)))
    ### END YOUR CODE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=20000)
    logisticR_classifier.fit_BGD(train_X, train_y, train_y.shape[0])
    #Evaluate
    test_X_all = prepare_X(raw_test_data)
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx] 
    test_y[np.where(test_y==2)] = -1

    print("Sigmoid classifier convergence weights: {}".format(logisticR_classifier.get_params()))
    print("Scores on training: {}, validation: {}, testing: {}".format(logisticR_classifier.score(train_X, train_y), logisticR_classifier.score(valid_X, valid_y), logisticR_classifier.score(test_X, test_y)))
    ### END YOUR CODE
    ################Compare and report the observations/prediction accuracy
    #It has been observed that both the models give same performance upon convergence
    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE
    #After each of the training steps 5, 10,.. weights have been outputted to show that the relation w_1 - w_2 = w holds
    #for every training step if we have learning rate of sigmoid = 2 * learning rate of softmax
    #Sigmoid classifier
    iterations_steps = [5, 10, 100, 200, 500]
    for iterations in iterations_steps:
        train_X = train_X_all[train_idx]
        train_y = train_y_all[train_idx]
        train_X = train_X[0:1350]
        train_y = train_y[0:1350]
        valid_X = valid_X_all[val_idx]
        valid_y = valid_y_all[val_idx]
        #####       set lables to -1 and 1 for sigmoid classifer
        train_y[np.where(train_y==2)] = -1
        valid_y[np.where(valid_y==2)] = -1

        logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=iterations)
        logisticR_classifier.fit_BGD(train_X, train_y, train_y.shape[0])
        weights = logisticR_classifier.get_params()
        #print("Weights for sigmoid for {} iterations: {}".format(iterations, weights))

        #Softmax classifier
        train_X = train_X_all[train_idx]
        train_y = train_y_all[train_idx]
        train_X = train_X[0:1350]
        train_y = train_y[0:1350]
        valid_X = valid_X_all[val_idx]
        valid_y = valid_y_all[val_idx]
        #Set labels to 0 and 1 for softmax classifier
        train_y[np.where(train_y == 1)] = 0
        valid_y[np.where(valid_y == 1)] = 0 
        train_y[np.where(train_y==2)] = 1
        valid_y[np.where(valid_y==2)] = 1

        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.25, max_iter=iterations,  k= 2)
        logisticR_classifier_multiclass.fit_BGD(train_X, train_y, train_y.shape[0])
        weights_multiclass = logisticR_classifier_multiclass.get_params()
        #print("Weights for softmax for {} iterations: {}".format(iterations, weights_multiclass))

    ### END YOUR CODE
    # ------------End------------

if __name__ == '__main__':
	main()
    
    
