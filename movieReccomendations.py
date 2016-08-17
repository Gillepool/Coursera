import scipy.io
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lamnda):

    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    J = 0
    X_grad = np.zeros((np.shape(X)))
    Theta_grad = np.zeros((np.shape(Theta)))

    J_temporary = (X.dot(Theta.T) - Y)**2
    J = np.sum(np.sum(J_temporary[R == 1]))/2 + Lamnda/2 * np.sum(np.sum(Theta**2)) + Lamnda/2 * np.sum(np.sum(X**2))

    X_grad = ((X.dot(Theta.T) - Y ) *R ).dot(Theta) + Lamnda*X
    Theta_grad = ((X.dot(Theta.T) - Y) * R).T.dot(X) + Lamnda*Theta

    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    return J, grad

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

def checkCostFunction(Lambda = None):
    if Lambda == None:
        Lambda = 0
    X_t = np.random.rand(4,3)
    theta_t = np.random.rand(5,3)

    Y = X_t.dot(theta_t.T)
    Y[np.random.rand(np.shape(Y)[0]) > 0.5] = 0
    R = np.zeros((np.shape(Y)))
    R[Y != 0] = 1

    m, n = np.shape(X_t)
    X = np.random.randn(m,n)
    a, b = np.shape(theta_t)
    theta = np.random.randn(a,b)
    num_users = np.shape(Y)[1]
    num_movies = np.shape(Y)[0]
    num_features = np.shape(theta_t)[1]
    def J(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, \
                                num_features, Lambda)

    numgrad = computeNumericalGradient(J, \
            np.append(X.flatten(), theta.flatten()))
    cost, grad = cofiCostFunc(np.append(X.flatten(), \
            theta.flatten()), Y, R, num_users, \
                          num_movies, num_features, Lambda)
    #print numgrad, grad
    #print 'The above two columns you get should be very similar.'
    #print '(Left-Your Numerical Gradient, Right-Analytical Gradient)'
    #diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    #print 'If your backpropagation implementation is correct, then \
           #the relative difference will be small (less than 1e-9).\
           #Relative Difference: ', diff



def LoadMovieList():
    counter = 0
    movielist = {}
    with open('data/movie_ids.txt', 'rb') as fid:
        lines = fid.readlines()
        for line in lines:
            movielist[counter] = line.split(' ', 1)[1]
            counter += 1
    return movielist

def normalizeRatings(Y, R):

    [m, n] = np.shape(Y)
    Ymean = np.zeros((m, 1))
    YNorm = np.zeros(np.shape(Y))
    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        YNorm[i, idx] = Y[i, idx] - Ymean[i]
    return YNorm, Ymean

print 'Loading movie ratings dataset.'
data = scipy.io.loadmat('data/ex8_movies.mat')
R = data['R']
Y = data['Y']
print 'Average rating for movie 1 (Toy Story): %8.8f/5 ' \
        %np.mean(Y[0,np.where(R[0,:] -1 == 0)])

plt.figure(figsize=(5, 5))
plt.imshow(Y)
#plt.show()

data1 = scipy.io.loadmat('data/ex8_movieParams.mat')
X = data1['X']
theta = data1['Theta']
# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[0:num_movies, 0:num_features]
theta = theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]


J, grad = cofiCostFunc(np.append(X.flatten(), theta.flatten()), Y, R, num_users, num_movies, num_features, 0)

print(J)

checkCostFunction(0)

J, grad = cofiCostFunc(np.append(X.flatten(), theta.flatten()), Y, R, num_users, num_movies, num_features, 1.5)

print(J)

checkCostFunction(1.5)



#Main code for learning movie recommendations and categories goes here......
movieList = LoadMovieList()

my_ratings = np.zeros((1682,1))

my_ratings[1] = 4
my_ratings[98] = 2
my_ratings[7] = 3
my_ratings[12]= 5
my_ratings[54] = 4
my_ratings[64] = 5
my_ratings[66] = 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[335] = 5
my_ratings[2] = 5
my_ratings[3] = 1
my_ratings[99] = 5
my_ratings[123]= 5
my_ratings[333] = 4
my_ratings[76] = 5
my_ratings[97] = 3
my_ratings[212] = 5
my_ratings[200] = 4
my_ratings[90] = 1
my_ratings[330] = 1

print("\n\nNew User Ratings:\n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated %d for %s\n", my_ratings[i], movieList[i-1])

print 'Training collaborative filtering...'
data = scipy.io.loadmat('data/ex8_movies.mat')
R = data['R']
Y = data['Y']

Y = np.append(my_ratings, Y, 1)
R = np.append((my_ratings != 0), R, 1)

[Ynorm, Ymean] = normalizeRatings(Y, R)

#Usefule values
num_users = np.shape(Y)[1]
num_movies = np.shape(Y)[0]
num_features = 10 # Antal "kategorier"

#set init params (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initital_parameters = np.append(X.flatten(), Theta.flatten())

Lambda = 10

cost = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda)[0]
grad = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda)[1]

theta = op.minimize(cost, np.append(X.flatten(), Theta.flatten()), method='CG', jac=grad, options={'disp':True, 'maxiter':50})

theta = theta.x

print(theta)

X = theta[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

p = X.dot(Theta.T)
my_predictions = p[:, 0]+Ymean.flatten()

movieList = LoadMovieList()

ix = my_predictions.argsort()[::-1]
print(R)
for i in range(len(movieList)):
    j = ix[i]
    print("Predicting rating %.1f for movie %s" % (my_predictions[j]/2, movieList[j]))
