import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.io


class MLP():
    def __init__(self,topology: "list of ints",learning_rate = 0.1):
        self.weights = np.array([None]*(len(topology)-1))
        self.topology = topology
        self.As = None
        self.Zs = None
        self.eta = learning_rate
       
        self.biases = np.ones(len(topology) -1)
        
        for i in range(len(topology) -1):
            self.weights[i] = np.random.rand(topology[i],topology[i+1])

    def feedforward(self, input_values):
        self.As = []
        self.Zs = []
        for bias, weight in zip(self.biases,self.weights):
            z = (np.dot(np.transpose(weight),input_values))+bias
            a = self.sigmoid(z)
            self.Zs.append(z)
            self.As.append(a)
            input_values = a
        return input_values
        

    # Cost function + derivative of it
    def cost_func(self,y):
        difference = self.As[len(self.As)-1] - y
        squared_sum = difference@np.transpose(difference)
        return ((1/(2*len(self.As[len(self.As)-1])))*squared_sum)

    def cost_func_prime(self,y):
        difference = self.As[len(self.As)-1] - y
        return ((1/len(self.As[len(self.As)-1]))*np.sum(difference))
        

    # Activation function + derivative of activation function
    def sigmoid(self,Z):
        return (1/(1+np.exp(-Z)))
    
    def sigmoid_prime(self, Z):
        return (self.sigmoid(Z)*(1- self.sigmoid(Z)))

    def update_weights(self,index,weight_gradient):
        self.weights[index] = self.weights[index] - self.eta * weight_gradient

    def train(self, epochs, input_layer, y):
        error = []
        for i in range(epochs):
            self.feedforward(input_layer)
            error.append(self.cost_func(y))
            self.backpropagate(y)
        # plot the error
        return error
    
    def train_with_sequence(self, X, y):
        error = []
        for i in range(len(X)):
            self.feedforward(X[i])
            error.append(self.cost_func(y[i]))
            self.backpropagate(y[i])
        return error

    # needs to be debugged 
    def backpropagate(self,y):
        up_to_last_z = self.cost_func_prime(y) * self.sigmoid_prime(self.Zs[len(self.Zs)-1])

        i = len(self.As) -1 
        derivatives_of_a_so_far = up_to_last_z
        # review this 
        while i >= 0:
            weight_gradient = np.transpose(self.As[i])*derivatives_of_a_so_far

            self.update_weights(i, weight_gradient)   
            # adding how related the current z is to the one in the previous layer
            if 0 == i:
                break

            w= self.weights[i]

            derivatives_of_a_so_far = w* derivatives_of_a_so_far

            ds_dz =  self.sigmoid_prime(self.Zs[i-1])
            ds_dz.shape = (len(self.Zs[i-1]),1)
            derivatives_of_a_so_far = derivatives_of_a_so_far.T@ ds_dz
            i = i-1



def main():
    # load the pictures
    mat = scipy.io.loadmat("ex4data1.mat")
    X = mat["X"]
    Y1 = mat["y"]
    # replace 10 with 0 because the data is prepared for octave and there is no 0, 10 was used instead
    Y1 = np.where(Y1==10, 0, Y1) 
    Y = []
    for i in range(len(Y1)):
        arr = np.zeros(10)
        arr[Y1[i][0]] = 1
        Y.append(arr)

    Y = np.array(Y)        

    # separate the data to training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # create the neural network
    network = MLP([400,50,30,10])
    error_progress = network.train_with_sequence(X_train,y_train)
    plt.plot(error_progress)
    plt.xlabel("Iterations")
    plt.ylabel("Error value")
    plt.show()



def nn_test():
    network = MLP([2,3,2,1])
    input_layer = np.array([0.4,0.82])
    y = 0
    network.feedforward(input_layer)
    error = network.cost_func(y)
    print(error)
    error_progress = network.train(1000,input_layer,y)
    plt.plot(error_progress)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
    network.feedforward(input_layer)
    error = network.cost_func(y)
    print(error)


def analytical_test():

    network = MLP([2,3,2,1])
    w1 = np.array([[0.2,-0.4],[0.73,0.13],[-0.31,-0.21]])
    w2 = np.array([[0.4,-0.67,0.37],[-0.09,-0.51,0.17]])
    w3 = np.array([0.97,-0.83])
    a1 = np.array([0.5,0.3])

    # second layer
    z1 = w1@a1
    # adding bias
    z1 = z1 +1
    a2 = network.sigmoid(z1)

    # third layer
    z2 = w2@a2
    # adding bias
    z2 = z2 + 1    
    a3 = network.sigmoid(z2)
    

    # output layer
    z3 = w3@a3
    # adding bias
    z3 = z3 + 1
    
    a4 = network.sigmoid(z3)

    y = 0
    
    print(a4*a4)

    # backpropagate 
    eta = 0.15
    # output - 1st hidden
    print(a3)
    grad1 = (2*(a4-y)*(network.sigmoid(a4)*(1-network.sigmoid(a4))))*a3
    print("Gradient1 ",grad1)
    
    # update w3 = w3 - eta*grad1
    print("Gradient1 * eta ",eta*grad1)
    w3 = w3 - (eta*grad1)
    print("Updated w3 ",w3)
    
    # backpropagate for w2




    
    

    
    # input_layer = np.array([0.5,0.32])
    # output_layer = network.feedforward(input_layer)
    # print("Result of forward pass is: ",output_layer)


if __name__ == "__main__":
    main()

        