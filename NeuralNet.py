from __future__ import division
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io
import scipy.optimize


input_layer_size = 400 # 20x20 input images of digits
hidden_layer_size = 25 # 25 hidden units(neurons)
num_labels = 10 # 10 labels/classes from number 1 through 10

print 'Loading and visualizing data'

data = scipy.io.loadmat('data/ex4data1.mat')
X = data['X']
y = data['y']
m = np.shape(X)[0]
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]
print (sel)
def displayData(X, example_width=None):
    if example_width == None:
        example_width = int(np.round(np.sqrt(np.shape(X)[1])))

    m, n = np.shape(X)
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    print(example_height)
    print(example_width)

    print(m)

    # Beteen images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad), \
                              pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch and get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            initial_x = pad + j * (example_height + pad)
            initial_y = pad + i * (example_width + pad)
            display_array[initial_x:initial_x + example_height, \
            initial_y:initial_y + example_width] = \
                X[curr_ex, :].reshape(example_height, example_width) \
                / max_val
            curr_ex += 1
        if curr_ex > m:
            break

    # Display image
    img = scipy.misc.toimage(display_array.T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='Greys_r')
    plt.show()

def sigmoid(z):
    g =  np.zeros((np.shape(z)))
    g = 1/(1+np.exp(-z))

    return g

def GradientSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


def NnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
    num_labels, X, y, Lambda):


    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)]. \
        reshape(hidden_layer_size, (input_layer_size + 1))
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]. \
        reshape(num_labels, (hidden_layer_size + 1))

    m = np.shape(X)[0]
    J = 0
    theta1_grad = np.zeros((np.shape(theta1)))
    theta2_grad = np.zeros((np.shape(theta2)))
    a1 = np.append(np.ones((m, 1)), X, 1)
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((m, 1)), a2, 1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)

    y_matrix = np.zeros((m, num_labels))

    for i in range(m):
        y_matrix[i, y[i] - 1] = 1

    J = -(1 / m) * (np.sum(np.sum(y_matrix * np.log(a3))) + \
        np.sum(np.sum((1 - y_matrix) * np.log(1 - a3)))) + \
        Lambda / (2 * m) * np.sum(np.sum(theta1[:, 1:] ** 2)) + \
        Lambda / (2 * m) * np.sum(np.sum(theta2[:, 1:] ** 2))

    d3 = a3 - y_matrix
    d2 = d3.dot(theta2) * a2 * (1 - a2)
    d2 = d2[:, 1:]
    theta1_grad = 1 / m * (d2.T.dot(a1) + Lambda * np.append( \
    np.zeros((hidden_layer_size, 1)), theta1[:, 1:], 1))
    theta2_grad = 1 / m * (d3.T.dot(a2) + Lambda * np.append( \
    np.zeros((num_labels, 1)), theta2[:, 1:], 1))
    grad = np.append(theta1_grad.flatten(), theta2_grad.flatten(), 0)

    return [J, grad]

def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1+L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init -epsilon_init
    return W

def debugInitWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.sin(np.arange(len(W.flatten()))+1).reshape(np.shape(W))/10

    return W

def computeNumericalGradient(J, theta):
    numgrad = np.zeros((np.shape(theta)))
    perturb = np.zeros((np.shape(theta)))
    e = 1e-4

    for p in range(len(theta.flatten())):
        perturb[p] = e
        loss1, grad1 = J(theta - perturb)
        loss2, grad2 = J(theta + perturb)

        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad

def checkNNGradient(Lambda=None):
    if Lambda == None:
        Lambda = 0

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = randInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = randInitializeWeights(num_labels, hidden_layer_size)

    X = debugInitWeights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(m) + 1, num_labels)
    nn_params = np.append(theta1.flatten(), theta2.flatten())

    def costFunc(p):
        cost, grad = NnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        return [cost, grad]

    cost, grad = costFunc(nn_params)

    numgrad = computeNumericalGradient(costFunc, nn_params)

    print(np.array([numgrad, grad]).T)
    print 'The above two columns should be very similair, (Left - numerical gradient, Right - Analytical gradient)'

    relative_diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print  'You relative difference should be very small (less than 1e-9)'
    print relative_diff

def predict(theta1, theta2, X):
    m = np.shape(X)[0]
    p = np.zeros((m ,1))

    h1 = sigmoid(np.append(np.ones((m, 1)), X, 1).dot(theta1.T))
    h2 = sigmoid(np.append(np.ones((m, 1)), h1, 1).dot(theta2.T))

    for i in range(np.shape(h2)[0]):
        p[i] = np.where(h2 == h2[i,:].max())[1]

    return p + 1

displayData(sel)
'''
print('Loading saved neural net parameters....')
data1 = scipy.io.loadmat('data/ex4weights.mat')
theta1 = data1['Theta1']
theta2 = data1['Theta2']

#Unrolling parameters
nn_params = np.append(theta1.flatten(), theta2.flatten(), axis=0)
print(nn_params)

print 'Feedforward using neural net...'

Lambda = 0
J, grad = NnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print 'Cost at parameters (loaded from ex4weights): (about 0.287629)', J

Lambda = 1
J, grad = NnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print 'Cost at parameters (loaded from ex4weights): (about 0.383759)', J

print("Evaluating gradient sigmoid...")
g = GradientSigmoid(np.array([-1,-0.5,0,0.5,1]))
print 'Gradient Sigmoid evaluated at [-1,-0.5,0,0.5,1]', g


print 'Checking Backproppagation'
Lambda = 3
checkNNGradient(Lambda)

debug_j, grad_j = NnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print 'Cost at (fixed) debugging parameters should be about, 0.576051: ', debug_j

'''

print 'Initializing Neural net parameters...(theta)'
init_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
init_theta2 = randInitializeWeights(hidden_layer_size, num_labels)

init_nn_params = np.append(init_theta1.flatten(), init_theta2.flatten())


print 'Training the neural net'
Lambda = 1


costFunc = lambda p: NnCostFunction(p, input_layer_size, hidden_layer_size,
                                    num_labels, X, y, Lambda)[0]
gradFunc = lambda p: NnCostFunction(p, input_layer_size, hidden_layer_size,
                                    num_labels, X, y, Lambda)[1]

#result2 = opt.minimize(fun=costFunc, x0=init_nn_params, \
#                    method='CG', args=(input_layer_size, hidden_layer_size, num_labels, X, y, Lambda), jac=gradFunc, options={'disp': True, 'maxiter': 100});
result2 =  opt.minimize(fun = costFunc, x0 = init_nn_params, \
                     method = 'CG', jac = gradFunc, options = {'disp': True, 'maxiter': 100});

nn_params = result2.x
cost = result2.fun

print("cost:", cost)


theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)]. \
    reshape(hidden_layer_size, (input_layer_size + 1))
theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]. \
    reshape(num_labels, (hidden_layer_size + 1))

print 'visualizing network'
displayData(theta1[:, 1:])

pred = predict(theta1, theta2, X)
print 'training accuracy: ', np.mean(np.where(pred == y, 1, 0))

#Animations of the drawing and the predition
from matplotlib.animation import FuncAnimation

def arrayToImage(i):
    x = np.reshape(X[i],(20,20)).T
    return x

def animate(self, *args):
    i = np.random.randint(5000)
    im = plt.imshow(arrayToImage(i), cmap='Greys_r')
    prediction.set_text("Prediction: %d"%pred[i])
    return im, prediction

fig = plt.figure()
ax=plt.axes()
im=ax.imshow(arrayToImage(0), cmap='Greys_r')
prediction = ax.text(.5, 1, '', color='white')

anim = FuncAnimation(fig, animate, interval=1500)
plt.show()

