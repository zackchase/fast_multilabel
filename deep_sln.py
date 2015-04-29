import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from scipy.special import expit
from load import mnist, yeast


srng = RandomStreams()

def rectify(X):
    return T.maximum(X, 0.)
    
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))
    
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    
def logit(X):
    return 1 / (1 + T.exp(-X))
    

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def model(X, w_ih1, w_io, w_h1h2, w_h1o, w_h2o, b_h1, b_h2, b_o, p_drop):
  
    flat_x = T.flatten(X, outdim=2)
    h1_in = T.dot(flat_x, w_ih1) + b_h1
    h1_activation = rectify(h1_in)
    h1_out = dropout(h1_activation, p_drop)
    
    h2_in = T.dot(h1_out, w_h1h2) + b_h2
    h2_activation = rectify(h2_in)
    h2_out = dropout(h2_activation, p_drop)
    
    
    output_in = T.dot(h2_out, w_h2o) + b_o #+  T.dot(h1_out, w_h1o)   + T.dot(flat_x, w_io)
                
    output_activation = logit(output_in)
    
    return output_activation


trX, teX, trY, teY = yeast(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

# trX = trX.reshape((-1, 784))
# teX = teX.reshape((-1, 784))

print "trX shape: "
print trX.shape 

print "trY.shape: "
print trY.shape

num_input   = trX.shape[1]
num_h1  = 500
num_h2 = 500
num_output  = trY.shape[1]

  
w_ih1   = init_weights((num_input, num_h1))
w_io    = init_weights((num_input, num_output))
w_h1o   = init_weights((num_h1, num_output))
w_h2o   = init_weights((num_h2, num_output))
w_h1h2  = init_weights((num_h1, num_h2))

b_h1 = theano.shared(floatX(np.zeros(num_h1)))
b_h2 = theano.shared(floatX(np.zeros(num_h2)))
b_o = theano.shared(floatX(np.zeros(num_output)))




noisy_y_hat = model(X, w_ih1, w_io, w_h1h2, w_h1o, w_h2o, b_h1, b_h2, b_o, .5)

dbf = theano.function([X],noisy_y_hat.shape, allow_input_downcast=True)
print "shape of noisy_y_hat"
print dbf(trX[0:128])



cost = -T.mean(T.log(noisy_y_hat) * Y + T.log(1-noisy_y_hat) * (1-Y) )

pred_y_hat = model(X, w_ih1, w_io, w_h1h2, w_h1o, w_h2o, b_h1, b_h2, b_o, 0.)

# v = T.vector()
# threshold = theano.function([v], T.switch(v < .5, 0., 1.))

pred_y = T.switch(pred_y_hat <.5, 0., 1.)

params = [w_ih1, w_h2o, w_h1h2, b_h1, b_h2, b_o]

updates = RMSprop(cost, params, lr=0.0001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=pred_y, allow_input_downcast=True)




for i in range(10000):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    # for i in range(len(trX)):
        cost = train(trX[start:end], trY[start:end])
        # cost = train(trX[i], trY[i])

    # print "cost function on train: "
    # print cost
    
    # print "test accuracy"
    print np.mean(np.equal(teY, predict(teX)))
    # print np.mean(np.argmax(teY, axis=1) == predict(teX))




    