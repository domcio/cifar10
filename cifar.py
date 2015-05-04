
# coding: utf-8

# In[ ]:

import numpy
import theano
import theano.tensor as T
import cPickle
import os

# theano.config.exception_verbosity = "high"
# theano.config.blas.ldflags="-L/usr/lib -lblas"


rng = numpy.random

os.path.abspath(os.path.curdir)
os.chdir('/home/dominik/dev/craftinity/cifar-10-batches-py')
paths = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

def unpickle(file):
	f = open(file, 'rb')
	dict = cPickle.load(f)
	f.close()
	return dict

def clear_data(data_dict):
    X = data_dict['data']
    Y = data_dict['labels']
    
    X = X.astype(int)
    X = X[:,0:1024] + X[:,1024:2048] + X[:,2048:3072]
    X = X.astype(float)
    X = X / 765.0
    
    YY = numpy.zeros((len(Y), 10))
    for i in xrange(len(Y)):
        YY[i, Y[i]] = 1
        
    return (X, YY)

train_sets = map(lambda x: clear_data(unpickle(x)), paths)        

(X, Y) = train_sets[0]


# In[ ]:

Y.shape


# In[ ]:

features = X.shape[1]
examples = X.shape[0]
classes = 10
middle_units1 = 100
middle_units2 = 50
iterations = 50
batch_size = 128
learning_rate = 1
momentum = 0.8


# In[ ]:

x = T.matrix("x")
y = T.matrix("y")

w1 = theano.shared(rng.randn(features, middle_units1)*0.01, name="w1")
w2 = theano.shared(rng.randn(middle_units1, middle_units2)*0.01, name="w2")
w3 = theano.shared(rng.randn(middle_units2, classes)*0.01, name="w3")
vb = theano.shared(rng.randn(1, middle_units1)*0.01, name="vb", broadcastable=(True, False))
hb1 = theano.shared(rng.randn(1, middle_units2)*0.01,  name="hb1", broadcastable=(True, False))
hb2 = theano.shared(rng.randn(1, classes)*0.01,  name="hb2", broadcastable=(True, False))

mw1 = theano.shared(numpy.zeros((features, middle_units1)), name="mw1")
mw2 = theano.shared(numpy.zeros((middle_units1, middle_units2)), name="mw2")
mw3 = theano.shared(numpy.zeros((middle_units2, classes)), name="mw3")
mvb = theano.shared(numpy.zeros((1, middle_units1)), name="mvb", broadcastable=(True, False))
mhb1 = theano.shared(numpy.zeros((1, middle_units2)),  name="mhb1", broadcastable=(True, False))
mhb2 = theano.shared(numpy.zeros((1, classes)),  name="mhb2", broadcastable=(True, False))


# In[ ]:

# [w1, w2, w3, vb, hb1, hb2, mw1, mw2, mw3, mvb, mhb1, mhb2] = \
# [theano.shared(numpy.load(x + ".dat"), name=x) for x in ['w1', 'w2', 'w3', 'vb', 'hb1', 'hb2', 'mw1', 'mw2', 'mw3' ,'mvb', 'mhb1', 'mhb2']]

[w1, w2, w3, mw1, mw2, mw3] = [theano.shared(numpy.load(x + ".dat"), name=x) for x in ['w1', 'w2', 'w3', 'mw1', 'mw2', 'mw3']]

[vb, hb1, hb2, mvb, mhb1, mhb2] = [theano.shared(numpy.load(x + ".dat"), name=x, broadcastable=(True, False)) for x in ['vb', 'hb1', 'hb2', 'mvb', 'mhb1', 'mhb2']]




# In[ ]:

hidden_act = 1 / (1 + T.exp(- T.dot(x, w1) - vb))

hidden2_act = 1 / (1 + T.exp(- T.dot(hidden_act, w2) - hb1))

output_act = - T.dot(hidden2_act, w3) - hb2

normalized_output = T.exp(output_act - output_act.max(axis=1, keepdims=True))

p = normalized_output / normalized_output.sum(axis=1, keepdims=True)

Jpt = - y * T.log(p)
    
cost = Jpt.mean()

gw1, gw2, gw3, gvb, ghb1, ghb2 = T.grad(cost, [w1, w2, w3, vb, hb1, hb2])

train = theano.function(
    inputs=[x, y],
    outputs=[p, cost],
    updates=((mw1, momentum*mw1 - learning_rate * gw1),
             (mw2, momentum*mw2 - learning_rate * gw2),
             (mw3, momentum*mw3 - learning_rate * gw3),
             (mvb, momentum*mvb - learning_rate * gvb),
             (mhb1, momentum*mhb1 - learning_rate * ghb1),
             (mhb2, momentum*mhb2 - learning_rate * ghb2),
             (w1, w1 + mw1),
             (w2, w2 + mw2),
             (w3, w3 + mw3),
             (vb, vb + mvb),
             (hb1, hb1 + mhb1),
             (hb2, hb2 + mhb2)))
predict = theano.function(inputs=[x], outputs=p)


# In[ ]:

for i in xrange(iterations):
    print 'iteration',i
    tot_err = 0
    for X, Y in train_sets:
        for start in xrange(0, X.shape[0], batch_size):
            batch_X = X[start:min(start + batch_size, X.shape[0]-1)]
            batch_Y = Y[start:min(start + batch_size, Y.shape[0]-1)]
            pred, err = train(batch_X, batch_Y)
            tot_err += err
    print 'error',tot_err


# In[ ]:

for mat in [w1, w3, vb, hb1, hb2, mw1, mw2, mw3, mvb, mhb1, mhb2]:
    mat.get_value().dump(mat.name + ".dat")


# In[ ]:

test_X, test_Y = clear_data(unpickle('test_batch'))


# In[ ]:

preds = T.matrix("preds")
true_classes = T.matrix("true")
true_classes_1 = T.argmax(true_classes, axis=1, keepdims=True)
predicted_class = T.argmax(preds, axis=1, keepdims=True)
accuracy = T.sum(T.eq(predicted_class, true_classes_1), dtype='float64') / true_classes.shape[0]
compare = theano.function(inputs=[preds, true_classes], outputs=accuracy)

get_most_probable_class = theano.function(inputs=[preds], outputs=predicted_class)

predictions = predict(test_X)
predicted_classes = get_most_probable_class(predictions)
acc = compare(predictions, test_Y)


# In[ ]:

print acc
print predicted_classes
predicted_classes.dump('prediction.dat')

