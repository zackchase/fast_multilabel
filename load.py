import numpy as np
import os
import cPickle
from scipy import linalg

datasets_dir = '../datasets/'

def preprocess(x,contrast="instance",center="instance"):
    if center=="instance":
        means = np.mean(x,axis=1)
        x -= means[:,np.newaxis]
    elif center =="feature":
        means = np.mean(x,axis=0)
        x -= means[np.newaxis,:]
    
    # if contrast=="instance":
    #     contrasts = np.std(x,axis=1)
    #     x /= contrasts[:,np.newaxis]
    # elif contrast =="feature":
    #     contrasts = np.std(x,axis=0)
    #     x /= contrasts[np.newaxis,:]
    
    
    return x

def zca_whitening_matrix(xtrain,regularization=10**-5):
    sigma = np.dot(xtrain.T,xtrain) / float(xtrain.shape[0])
    U, S, V = linalg.svd(sigma)
    tmp = np.dot(U, np.diag(1/np.sqrt(S+regularization)))
    zca_mat = np.dot(tmp, U.T)
    return zca_mat
    

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def cifar(ntrain=50000, ntest= 10000, onehot=True):
  data_dir = os.path.join(datasets_dir, 'cifar-10')
  
  
  trX = np.array([])
  trTempY = []
  trY = np.zeros(500000).reshape(50000,10)
  
  teX = np.array([])
  teTempY = []
  teY = np.zeros(100000).reshape(10000,10)
  
  for i in range(1,6):
    train_dict = unpickle(os.path.join(data_dir, 'data_batch_' + str(i)))
    trX = np.append( trX, train_dict['data'])
    trTempY = trTempY + train_dict['labels']

  for i in range(len(trTempY)):
    trY[i][trTempY[i]] = 1.

  test_dict = unpickle(os.path.join(data_dir, 'test_batch'))
  teX = test_dict['data']
  teTempY = test_dict['labels']
  
  for i in range(len(teTempY)):
    teY[i][teTempY[i]] = 1.
  
  # trX = preprocess(trX.reshape((50000,3072)))
  # teX = preprocess(teX.reshape((10000, 3072)))
  

  # zca_mat = zca_whitening_matrix(trX)
  
  # trX = np.dot(trX, zca_mat.T)
  # teX = np.dot(teX, zca_mat.T)
  
  return trX.reshape((50000,3072)), teX.reshape((10000,3072)), trY, teY
  #return tempY
  
  
def yeast(ntrain=1500, ntest=917, onehot=True):

  trX = np.zeros((1500, 103))
  trY = np.zeros((1500, 14))
  
  teX = np.zeros((917, 103))
  teY = np.zeros((917, 14))

  data_dir = os.path.join(datasets_dir,'yeast/')
  fd = open(os.path.join(data_dir,'yeast_train.svm'))  
  lines = fd.readlines()

  fd.close()  
  
  for i in range(len(lines)):
    line = lines[i]
    toks = line.strip().split(" ")
    labels = toks[0].split(",")

    for label in labels:
      trY[i,int(label)-1] = 1.
      #print int(label)
         
      features = toks[1:]
      for feature in features:
        j, val = feature.split(":") 
        trX[i,int(j)-1] = float(val)
        
  fd = open(os.path.join(data_dir,'yeast_test.svm'))  
  lines = fd.readlines()
  fd.close()     
       
  for i in range(len(lines)):
    line = lines[i]
    toks = line.strip().split(" ")
    labels = toks[0].split(",")

    for label in labels:
      teY[i,int(label)-1] = 1.
      #print int(label)
         
      features = toks[1:]
      for feature in features:
        j, val = feature.split(":") 
        teX[i,int(j)-1] = float(val)
      
  return trX, teX, trY, teY
      
  

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY