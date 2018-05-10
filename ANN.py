import preprocessing_for_sea as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np

class ANNModel:

    def __init__(self):
        self.X = None
        self.Y = None

    def create_placeholders(self,n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.

        Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)

        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

        Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
          In fact, the number of examples during test/train is different.
        """

        self.X = tf.placeholder(tf.float32, [n_x, None], name="x")
        self.Y = tf.placeholder(tf.int64, [None], name="y")

        return self.X, self.Y


    def initialize_parameters(self,n_x,n_y):
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                            W1 : [2, n_x]
                            b1 : [2, 1]
                            W2 : [2, 2]
                            b2 : [2, 1]
                            W3 : [n_y,2]
                            b3 : [n_y,1]

        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """

        tf.set_random_seed(1)  # so that your "random" numbers match ours
        W1 = tf.get_variable("W1", [2, n_x], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable("b1", [2, 1], initializer=tf.zeros_initializer())
        W2 = tf.get_variable("W2", [2, 2], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable("b2", [2, 1], initializer=tf.zeros_initializer())
        W3 = tf.get_variable("W3", [n_y, 2], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer())


        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}

        return parameters


    def forward_propagation(self,X, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                      the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """

        # Retrieve the parameters from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']


        Z1 = tf.matmul(W1, X) + b1  # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
        Z2 = tf.matmul(W2, A1) + b2  # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
        Z3 = tf.matmul(W3, A2) + b3  # Z3 = np.dot(W3,Z2) + b3


        return Z3, Z2


    def compute_cost(self,Z3, Y):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3

        Returns:
        cost - Tensor of the cost function
        """

        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))

        return cost


    def model(self,batchdata,learning_rate=0.001, num_epochs=25000, Train=True):
        """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

        Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)  # to keep consistent results
        seed = 3  # to keep consistent results
        m, n_x = batchdata[0]['xtrain'].shape # (n_x: input size, m : number of examples in the train set)
        n_y = 2 # n_y : output size
        costs = []  # To keep track of the cost
        out_prob_collect = []
        # Create Placeholders of shape (n_x, n_y)
        X, Y = self.create_placeholders(n_x, n_y)

        # Initialize parameters
        parameters = self.initialize_parameters(n_x,n_y)


        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3,Z2 = self.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop

            for epoch in range(num_epochs):
                seed = seed + 1
                # IMPORTANT: The line that runs the graph on a minibatch.

                sess.run((optimizer), feed_dict={X: batchdata[0]['xtrain'].transpose(),Y: batchdata[0]['ytrain']})

                if epoch % 1000 == 0:
                    minibatch_cost = sess.run(cost,feed_dict={X: batchdata[0]['xtest'].transpose(),Y: batchdata[0]['ytest']})


                    # Calculate the correct predictions
                    Y_pred = tf.argmax(Z3)
                    correct_prediction = tf.equal(Y_pred, Y)

                    # Calculate accuracy on the test set
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    # print ("Train Accuracy:", accuracy.eval({X: batchdata['xtrain'].transpose(),Y: batchdata['ytrain']}))
                    # print ("Test Accuracy:", accuracy.eval({X: batchdata['xtest'].transpose(),Y: batchdata['ytest']}))

            # plot the cost
            # plt.plot(np.squeeze(costs))
            # plt.ylabel('cost')
            # plt.xlabel('iterations (per tens)')
            # plt.title("Learning rate =" + str(learning_rate))
            # plt.show()

            # lets save the parameters in a variable
            # Z3_prob = tf.nn.softmax(tf.transpose(Z3))
            # plt.plot(out_prob)
            # plt.show()
            print ("Parameters have been trained!")


            # predict decision boundaries
            # build static plot
            x = batchdata[0]["xtest"]
            y = batchdata[0]["ytest"]

            # build mesh for decision boundaries
            h = 0.05
            x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
            y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            tmp = np.c_[xx.ravel(), yy.ravel()]
            tmp2 = np.zeros((tmp.shape[0], 1))
            input_data = np.concatenate((tmp,tmp2), axis=1)
            y_pred = sess.run(Y_pred,feed_dict={X:np.transpose(input_data)})

            # transform features
            for timestep in range(0,len(batchdata),1):
                minibatch_prob = sess.run(Z2,feed_dict={X: batchdata[timestep]['xtest'].transpose(),Y: batchdata[timestep]['ytest']})
                out_prob_collect.append(minibatch_prob.transpose())
            return out_prob_collect, (xx, yy, y_pred)


    def instance_learning(self,batchdata,learning_rate=0.001, num_epochs=25000, Train=True):
        """

        :param batchdata:
        :param learning_rate:
        :param num_epochs:
        :param Train:
        :return:
        """
        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)  # to keep consistent results
        seed = 3  # to keep consistent results
        costs = []  # To keep track of the cost
        n_x = batchdata["xtrain"].shape[1]
        n_y = 2
        X, Y = self.create_placeholders(n_x,n_y)
        parameters = self.initialize_parameters(n_x,n_y)
        Z = self.forward_propagation(X,parameters)
        cost = self.compute_cost(Z,Y)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for samples in range(batchdata["xtrain"].shape[0]):
                tmp_x = batchdata["xtrain"][samples,:].transpose()
                tmp_y = batchdata["ytrain"][samples]
                _, tmp = sess.run([optimizer, cost], feed_dict={X: batchdata["xtrain"][samples,:].reshape((n_x,1)),
                                                                Y: batchdata["ytrain"][samples].reshape(1)})
                # print tmp

                if samples % 1000 == 0:
                    correct_prediction = tf.equal(tf.argmax(Z), Y)

                    # Calculate accuracy on the test set
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                    print ("Test Accuracy:", accuracy.eval({X: batchdata["xtest"].transpose(),Y: batchdata["ytest"]}))

            parameters = sess.run(parameters)
            return parameters, None
