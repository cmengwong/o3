import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util imiport relu, error_rate, getKaggleMNIST, init_weights

# an_id -  setting the name of theano id
class AutoEncoder(object):
    def __init__(self, M, an_id):
        self.M = M
        self.id = an_id
    def fit(self, X, learning_rate = 0.5, mu = 0.99, epochs = 1, batch_sz = 100, show_figure = False):
        N, D = X.shape()
        n_batches = N / batch_sz
        W0 = init_weights( D, self_M)
        self.W = theano.shared( W0 ,'W_%s' % self_id)
        self.bh = theano.shared( np.zeros(self.M), 'bh_%s' % self.id)
        self.bo = theano.shared( np.zeros(D), 'bo_%s' % self.id)

        self.params = [self.W, self.bh, self.bo]
        self.forward_params = [self.W, self.bh]

        # use for caculating the di
        self.dW= theano.shared( np.zeros(W0.shape), 'dW_%s' % self.id)
        self.dbh = theano.shared( np.zeros(self.M), 'dbh_%s' % self.id)
        self.dbo = theano.shared( np.zeros(D), 'dbo_%s' % self.id)
        self.dparams = [self.dW, self.dbh, self.dbo]
        self.dforward_params = [self.dW, self.dbh]

        # define the tensor input
        X_in = T.martix( 'X_%s' % self.id)
        X_hat = self.forward_output(X_in)

        # next define the hidden layer operation as a theano function
        H = T.nnet.sigmod(X_in.dot(self.W)+ self.bh)
        self.hidden_op = theano.function(
            inputs=[X_in],
            outputs=H,
        )
        # define the class function
        cost = ((X_in - X_hat)*(X_in - X_hat)).sum() / N
        # cost = -(X_in * T.log(X_hat) + (1 - X_in)*(T.log(1 - X_hat))).sum() / N
        cost_op = theano.function(
            inputs=[X_in],
            outputs=cost,
        )

        # define my update
        updates = [
            (p, p + mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
        ] + [
            (dp, mu*dp -learning_rate*T.grad(cost, p)) for p,dp in zip(self.params, self.dparams)
        ]

        train_op = theano.function(
            inputs=[X_in],
            updates=updates,
        )

        # for saving the cost in every iteration
        costs = []
        print("training autoencoder: %s" % self.id)
        for i in range(epochs):
            print("epochs:",i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = j*batch_sz : (j+1) * batch_sz
                train_op(batch)
                the_cost = cost_op(X)
                print("j / n_batches:", j, "/", n_batches, "cost:", the_cost)
                costs.append(the_cost)
        if show_figure:
            plt.plot(costs)
            plt.show()
        def forward_hidden(self, X):
            Z = T.nnet.sigmod(X.dot(self.W)+ self.bh)
            return Z
        def forward_output(self, X):
            Z = self.forward_hidden(X)
            Y = T.nnet.sigmod(Z.dot(self.W.T) + self.bo)
            return Y

class DNN(object):
    def __init__(self, hidden_layer_sizes, UnsupervisedModel = AutoEncoder):
        self.hidden_laryer = []
        # id for each autoencoder
        count = 0
        for M in hidden_layer_sizes:
            ae = UnsupervisedModel(M, count)
            self.hidden_layer.append(ae)
            count +=1

    def fit(self, X, Y, Xtest, Ytest, pretrain = True, learning_rate = 0.01, mu = 0.99, epochs=1, reg  =0.1, batch_sz = 100):
        pretrain_epochs = 1
        if not pretrain:
            pretrain_epochs = 0

        current_input = X
        for ae in self.hidden_layers:
            # if pretrain_epochs is 0 -> there is not gonna do anything expect initalizing
            ae.fit(current_input, epcohs = pretrain_epochs)
            current_input = ae.hidden_op(current_input)

        #initalizing logistic regression stuff
        N = len(Y)
        K = len(set(Y))
        W0 = init_weights((self.hidden_layers[-1].M, K))
        self.W = theano.shared(W0,'W_logreg')
        self.b = theano.shared(np.zeros(K), 'b_logreg')

        # add the params to do the GD
        self.params = [self.W, self.b]
        # we have to add the others hidden layre to the params
        for ae in self.hidden_layers:
            self.params += ae.forward_params

        self.dW= theano.shared(np.zeros(W0.shape),'dW_logreg')
        self.db = theano.shared(np.zeros(K), 'db_logreg')

        for ae in self.hidden_layers:
            self.dparams += ae.forward_dparams

        #define input and output
        X_in = T.martrix('X_in')
        targers = T.ivector9('Targets')
        pY = self.forward(X_in)

        # define the regulization class
        squared_magnitude = [(p*p for p in self.params)]
        reg_cost = T.sum(squared.magnitude)
        # choose the elements which target is 1
        cost = -T.mean(T.log(pY[T.arange(pY.shape[0], targets])) + reg_cost)
        prediction = self.predict(X_in)
        cost_predict_op = theano.function(
            inputs = [X_in, targets]
            outputs = [cost, prediction]
        )

        updates = [
            (p, p + mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
        ] + [
            (dp, mu*dp -learning_rate*T.grad(cost, p)) for p,dp in zip(self.params, self.dparams)
        ]

        train_op = theano.function(
            inputs=[X_in, targets],
            updates=updates,
        )

        n_batches = N / batch_sz
        cost = []
        print("supervised training")
        for i in range(epochs)
            print("epoch: ",i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz : (j+1)*batch_sz]
                Ybatch = Y[j*batch_sz : (j+1)*batch_sz]
                train_op(Xbatch, Ybatch)
                the_cost, the prediction = cost_predict_op(Xtest, Ytest)
                error = error_rate(the_prediction, Ytest)
                print("j / n_batches:", j, "/", n_batches, "cost:", the_cost)
                costs.append(the_cost)
        plt.plot(costs)
        plot.show()

    def predict(slef, X):
        return T.argmax(self.forward(X), axis = 1)

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z
        Y = T.nnet.softmax(T.dot(current_input, self.W) + self.b)
        return Y

# define main function
    def main():
        Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
        dnn = DNN([1000, 750, 500])
        dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3)
if __name__ == '__main__'
    main()
