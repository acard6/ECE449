import hw5_utils as utils
import numpy as np
import torch
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n, _ = x_train.shape
    alpha = torch.zeros(n, requires_grad=True)
    K = torch.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(x_train[i], x_train[j])

    if c==None:
        alpha.clamp(min=0)
    else:
        alpha.clamp(min=0, max=c)

    for t in range(num_iters):
        grad = torch.ones(n)
        for i in range(n):
            grad[i] =0.5* y_train[i] * (alpha*y_train* K[:,i].t()).sum()

            alpha = alpha - lr*grad
            if c==None:
                alpha.clamp(min=0)
            else:
                alpha.clamp(min=0, max=c)

    return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    pass

def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    # Add column for bias 
    X = torch.cat((torch.ones(n, 1), X), dim=1)
    w = torch.zeros(d+1, 1) # w=0
    for i in range(num_iter):
        # Compute the gd of R(w)
        grad = (-X.t()@Y)@(1/(1 + torch.exp(-Y.t()@(X@w))))
        #b=(1 /(1 + torch.exp(-Y@(w.t()@X.t())))) 
        #grad = (X.t()@b@Y)
        w -= lrate * grad / n       # Update w
        
    return w

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass

if __name__ == "__main__":
    X, Y = utils.load_logistic_data()
    #X = torch.rand(5,3)
    #Y = torch.tensor([[1,-1,-1,1,-1]])
    #w = logistic(X,Y,num_iter=20)
    #print(w)
    alpha = svm_solver(X, Y, 0.0001, 20)
    print(alpha)
    