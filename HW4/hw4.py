import torch
import hw4_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    # Add column for bias 
    X = torch.cat((torch.ones(n, 1), X), dim=1)
    w = torch.zeros(d+1, 1) # w=0
    for i in range(num_iter):
        # Compute the gd of R(w)
        grad = torch.matmul(torch.matmul(X.t(), X), w) - torch.matmul(X.t(), Y)
        w -= lrate * grad / n       # Update w
    
    return w

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    ones = torch.ones(X.shape[0], 1)
    X = torch.cat([ones, X], dim=1)
    X_pinv = torch.pinverse(X)
    w = torch.matmul(X_pinv, Y)
    
    return w

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''

def plot_linear():
    # Load data
    X, Y = utils.load_reg_data()

    # Compute optimal parameters
    w = linear_normal(X, Y)

    # Plot data and line
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, w[0] + w[1] * X, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Lin Reg Plot')
    plt.show()

    return plt.gcf()

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X_squared = torch.zeros((n, 1+d+(d*(d+1)//2)))
    X_squared[:, 0] = 1
    curr_col = 1
    for i in range(d):
        for j in range(i, d):
            X_squared[:, curr_col] = X[:, i] * X[:, j]
            curr_col += 1

    w = torch.zeros(1+d+(d*(d+1)//2))
    for i in range(num_iter):
        scores = torch.matmul(X_squared, w)
        grad = torch.matmul(X_squared.T, (scores - Y.squeeze())) /(n)
        w -= lrate * grad

    return torch.cat((w[:1], w[1:d+1], w[d+1:]), dim=0)

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X_squared = []
    for i in range(d):
        for j in range(i, d):
            X_squared.append(X[:, i] * X[:, j])

    X_squared = torch.stack(X_squared, dim=1)
    X_combined = torch.cat([torch.ones((n, 1)), X, X_squared], dim=1)
    pinv = torch.pinverse(X_combined)
    w = pinv @ Y

    return w

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()

    # Generate curve using poly_normal()
    w = poly_normal(X, Y)
    curve_x = torch.linspace(0, 4, 100).reshape(-1, 1)
    curve_y = torch.cat([torch.ones((100, 1)), curve_x, curve_x**2], dim=1) @ w

    # Plot the curve and the data
    plt.scatter(X, Y, color='blue')
    plt.plot(curve_x, curve_y, color='red', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pol Reg Plot')
    plt.show()

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    # Load XOR data set
    X, Y = utils.load_xor_data()

    # Linear model
    linear_model = linear_normal(X, Y)

    # Polynomial model
    poly_model = poly_normal(X, Y)

    # Plot the results
    xmin, xmax, ymin, ymax = -1.1, 1.1, -1.1, 1.1


    print("Linear model predictions:")
    utils.contour_plot(xmin, xmax, ymin, ymax, linear_model)

    print("Polynomial model predictions:")
    utils.contour_plot(xmin, xmax, ymin, ymax, poly_model)

    return linear_model, poly_model


if __name__ == "__main__":
    poly_xor()
#    plot_linear()
#    X, Y = utils.load_reg_data()
#    w = poly_gd(X,Y,num_iter=20)
#    print(w)