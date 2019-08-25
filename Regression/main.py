import pandas as pd
import numpy as np


def mean_normalize(X):
    # z = (x - mu)/s where mu is the mean and s is the standard deviation = max- min value
    mu_hist = []
    sigma_hist=[]
    for i in range(1,len(X[0])):
        mu = np.mean(X[:,i])
        mu_hist.append(mu)
        s = max(X[:,i]) - min(X[:,i])
        sigma_hist.append(s)
        X[:,i] = (X[:,i]-mu)/s

    return [mu_hist,sigma_hist]
    

# X, y, theta- parameter to minimize, _lambda- regularization parameter
def cost_function(X,y,theta,_lambda):
    m = len(y)
    hypothesis = X@theta
    theta_without_0=theta[1:]
    diff = hypothesis-y

    diff_squared_sum = diff.transpose()@diff
    theta_squared_sum = theta_without_0.transpose()@theta_without_0


    # Regularization
    # J = 1/m * (H-y)^2 * lambda/2m * theta^2
    # (H-y)^2 = (H-y)' * (H-y)
    
    J = (1/m)* ((diff_squared_sum)+((_lambda/(2*m))*theta_squared_sum))

    # regularization term exclude theta[0] because for x0 there is no multiplied theta, consequently
    # derivative of const = 0

    theta_gradient = np.append([0],theta_without_0)    
    derivative = X.transpose()@diff
    derivative = (1/m)*derivative

    add_to_derivative = (_lambda/m)*theta_gradient
    add_to_derivative= add_to_derivative.reshape(derivative.shape)   

    grad = derivative + add_to_derivative

    return [J,grad]


# Thetha, eta - learning rate, calculated gradient
# calculated gradient = first derivative with respect to theta
def gradient_descent_step(theta, eta,gradient):
    change = eta * gradient
    theta = theta - change
    return theta

def gradient_descent(X,y,theta,eta,_lambda, iterations):
    for i in range(iterations):
        result = cost_function(X,y,theta,_lambda)

        theta = gradient_descent_step(theta,eta,result[1])

    return theta



# Normal equation
def calculate_theta_analytically(X,y):
    
    theta = X.transpose()@X
    second_product = X.transpose()@y
    theta = np.linalg.inv(theta)@second_product
    return theta


def main():
    data = np.genfromtxt("ex1data2.txt",delimiter=',')

    # separate the features
    X = data[:,:-1]
    Y = data[:,-1]
    y = Y.reshape((len(Y),1))

    # add X0 = 1
    first_column = np.ones((len(X),1))
    X = np.append(first_column,X,axis=1)
    
    # generate random theta
    theta = np.random.rand(len(X[0]),1)     

    alpha = 0.15
    num_iters = 20000
    [mu,sigma]=mean_normalize(X)
    theta = gradient_descent(X,y,theta,alpha,1,num_iters)
    print("Theta after gradient")
    print(theta)

    print("Calculated from normal equation")
    theta_norm = calculate_theta_analytically(X,y)
    print(theta_norm)

    # predict

    sqFt = (1650 - mu[0])/sigma[0]
    bedrooms = (3 - mu[1])/sigma[1]
    parameters = np.array([1, sqFt, bedrooms])
    price_norm = parameters@theta_norm
    print("Price of norm equation",price_norm)
    price = parameters@theta
    print("Price with gradient",price)

    


if __name__ == "__main__":
    main()