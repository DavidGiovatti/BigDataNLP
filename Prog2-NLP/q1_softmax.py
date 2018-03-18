import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### MY CODE:

    m = np.max (x, axis = len (x.shape) - 1, keepdims = True)
    e = np.exp (x - m)
    s = np.sum (e, axis = len (x.shape) - 1, keepdims = True)
    x = e / s

    ### END MY CODE 
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### MY CODE:

    test1 = softmax(np.array([1,4]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.04742587, 0.95257413]))) <= 1e-6

    test2 = softmax(np.array([[5,8],[101,104]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.04742587, 0.95257413], [0.04742587, 0.95257413]]))) <= 1e-6

    test3 = softmax(np.array([[-101,-104]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.95257413, 0.04742587]))) <= 1e-6

    print "Tests end..."


    ### END MY CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
