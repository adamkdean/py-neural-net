# deps
# settings
# nonlin function
# input
# output
# set layer weight (syn0)
# loop
#   set l0 to input  } \
#   set l1 to l0 dot }  - forward propagation
#   calculate error
#   calculate delta
#   update syn0
# print l1 at end

import numpy as np

np.random.seed(1)
np.set_printoptions(suppress=True)

def deriv(x):
    return x * (1 - x)
    
def nonlin(x):
    return 1 / (1 + np.exp(-x))
    
X = np.array([[0, 1],
              [1, 1],
              [1, 0]])

y = np.array([[0],
              [1],
              [1]])

syn0 = 2 * np.random.random((2, 1)) - 1

for i in xrange(50000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    
    l1_error = y - l1
    l1_delta = l1_error * deriv(l1)

    syn0 += np.dot(l0.T, l1_delta)
    
print "Result:", l1