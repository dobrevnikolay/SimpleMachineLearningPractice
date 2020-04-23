import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def reverse_enum(L):
   for index in reversed(range(len(L))):
      yield index, L[index]


class Layer:

    def __init__(self,input_units, output_units):
        pass
    def forward(self,x):
        pass
    def backward(self,x,gradient_output):
        pass

class Loss:
    
    def __init__(self):
        pass
    @staticmethod
    def compute_loss(y,target):
        pass
    @staticmethod
    def compute_gradient(y,target):
        pass


class CrossEntropy(Loss):
    def __init__(self):
        pass
    
    @staticmethod
    def _softmax(x):
        exps = np.exp(x)
        exp_sum = np.sum(exps,axis=-1)
        difference = exps/exp_sum[:,None]
        return difference

    @staticmethod
    def compute_loss(y, target):
        p = CrossEntropy._softmax(y)
        log_likelihood = -np.log(p[np.arange(len(y)),target])
        loss = np.sum(log_likelihood) / target.shape[0]
        return loss
    
    @staticmethod
    def compute_gradient(y, target):
        grad = CrossEntropy._softmax(y)
        grad[np.arange(len(y)),target] -= 1
        grad = grad/target.shape[0]
        return grad

class SquaredDifference(Loss):
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(y, target):
        return 0.5 * ((y-target)**2)

    @staticmethod
    def compute_gradient(y, target):
        return (y-target)



class Dense(Layer):
    def __init__(self,input_units, output_units, learning_rate = 0.01):
        self.w = np.random.normal(0,np.sqrt(2/input_units),size=(input_units,output_units))
        self.b = np.random.normal(0,np.sqrt(2/input_units),size=(1,output_units))
        self.lr = learning_rate

    def forward(self,x):
        x = np.dot(x,self.w)+self.b
        return x

    def backward(self,x,gradient_output):
        # x@w + b
        # dz/dx = w
        # dz/dw = x
        # dz/db = 1
        grad_input = np.dot(gradient_output,self.w.T)
        grad_w = np.dot(x.T,gradient_output)
        grad_b = np.sum(gradient_output,axis=0)

        # now we update the weights and biases

        self.w = self.w - self.lr * grad_w
        self.b = self.b - self.lr * grad_b.T
        
        return grad_input


class Dense_AdaGrad(Dense):
    def __init__(self, input_units, output_units, learning_rate=0.01):
        super().__init__(input_units, output_units, learning_rate=learning_rate)
        self.G_w = np.zeros_like(self.w)
        self.G_b = np.zeros_like(self.b)

    def backward(self, x, gradient_output):
        # x@w + b
        # dz/dx = w
        # dz/dw = x
        # dz/db = 1
        grad_input = gradient_output@self.w.T
        grad_w = x.T@gradient_output
        grad_b = np.sum(gradient_output,axis=0)

        # Now use AdaGrad optimization strategy by keeping the momentum
        self.G_w = self.G_w + grad_w**2
        self.w = self.w - ((self.lr/np.sqrt(self.G_w + 1e-5)) * grad_w)

        self.G_b = self.G_b + grad_b.T**2
        self.b = self.b - ((self.lr/np.sqrt(self.G_b + 1e-5)) * grad_b.T)

        return grad_input

class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self,x):
        return 1/(1+np.exp(-x))

    def backward(self, x, gradient_output):
        grad = self.forward(x) * ( 1 - self.forward(x))
        gradient_output = gradient_output*grad
        return gradient_output


class Tanh(Layer):
    def __init__(self):
        pass
    def forward(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def backward(self, x, gradient_output):
        grad = 1 - self.forward(x)**2
        gradient_output = gradient_output*grad
        return gradient_output

    
class ReLU(Layer):
    def __init__(self):
        pass
    def forward(self, x):
        return np.maximum(0,x)
    
    def backward(self, x, gradient_output):
        gradient_output = gradient_output*(x>0).astype(int)
        return gradient_output


class NeuronNetwork:

    def __init__(self, layers,activations, loss_function:"Loss", optimization, learning_rate):
        self.layers = []
        if( "AdaGrad" is optimization):
            for i in range(len(layers)-1):
                self.layers.append(Dense_AdaGrad(layers[i],layers[i+1],learning_rate))
                if( i < len(activations)):
                    self.layers.append(self._create_non_linearity(activations[i]))
        else:
            for i in range(len(layers)-1):
                self.layers.append(Dense(layers[i],layers[i+1],learning_rate))
                if( i < len(activations)):
                    self.layers.append(self._create_non_linearity(activations[i]))
        self.loss_function = loss_function

    def _create_non_linearity(self, non_linearity:"string"):

        non_linearity = non_linearity.lower()

        if "sigmoid" == non_linearity:
            return Sigmoid()
        elif "tanh" == non_linearity:
            return Tanh()
        elif "relu" == non_linearity:
            return ReLU()

    def forward(self,x):
        activations = []
        input = x
        for layer in self.layers:
            input = layer.forward(input)
            activations.append(input)

        return activations
    
    def predict(self,x):
        logits =  self.forward(x)[-1]
        return np.argmax(logits,axis=-1)
    
    def train(self, x, target):
        # make a forward pass
        activations = self.forward(x)

        layer_inputs =[x] + activations

        y = activations[-1]

        loss = self.loss_function.compute_loss(y,target)
        grad = self.loss_function.compute_gradient(y,target)

        for i, layer in reverse_enum(self.layers):
            grad = layer.backward(layer_inputs[i],grad)

        if x.shape[0] > 1:
            return np.mean(loss)
        else:
            return loss


(x_train,y_train) ,(x_test,y_test) = mnist.load_data()


x_train = x_train.astype(float) / 255.
x_test = x_test.astype(float) / 255.


x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])


plt.figure(figsize=[6,6])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title("Label: %i"%y_train[i])
    plt.imshow(x_train[i].reshape([28,28]),cmap='gray')

plt.show()

network = NeuronNetwork([x_train.shape[1],100,200,10],["relu","relu"],CrossEntropy,"AdaGrad",0.01)

epochs = 10

batch_size = 32

begin = 0
loss = 0
for _ in range(epochs):
    begin = 0
    while begin < x_train.shape[0]:
        batch = None
        target = None
        if(begin + batch_size > x_train.shape[0]):
            batch = x_train[begin:]
            target = y_train[begin:]
        else:
            batch = x_train[begin:begin+batch_size]
            target = y_train[begin:begin+batch_size]
            
        begin +=batch_size

        loss = network.train(batch,target)
    
    predicted_train = network.predict(x_train)
    predicted_test = network.predict(x_test)
    print("Epoch %d: Loss %.3f Train Accuracy %.3f  Test Accuracy %.3f "%(_,loss,np.mean(predicted_train==y_train),np.mean(predicted_test==y_test)))

