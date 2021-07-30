# Data-Science-Liniar-Regression

Subject
Getting and analysing existing data is the very first task of a data scientist.
The next step is to find tendencies and to generalize.

For example, let's say we want to know what a cat is. We can learn by heart some pictures of cats and then classify as cat animals that are similar to the pictures.
We then need a way to "measure" similarity. This is called instance-based learning.

Another way of generalizing is by creating a model from the existing examples and make prediction based on that model.

For instance, let's say we want to analyze the relation between two attributes and plot one against the other:


We clearly see a trend here, eventhough the data is quite noisy, it looks like the feature 2 goes up linearly as the feature 1 increases.
So in a model selection step, we can decide to go for a linear model.

feature_2 = θ0 + θ1 . feature_1

This model has two parameters, θ0 and θ1. After choosing the right values for them, we can make our model represent a linear function matching the data:


Everything stands in "choosing the right values". The "right values" are those for which our model performs "best".
We then need to define a performance measure (how well the model performs) or a cost function (how bad the model performs).

These kind of problems and models are called Linear Regression.
The goal of this journey is to explore linear and logistic regressions.

Introduction
A linear model makes predictions by computing a weighted sum of the features (plus a constant term called the bias term):

y = hθ(x) = θT·x = θnxn + ... + θ2x2 + θ1x1 + θ0

• y is the predicted value.
• n is the number of features.
• xi is the ith feature value (with x0 always equals to 1).
• θj is the jth model feature weight (including the bias term θ0).
• · is the dot product.
• hθ is called the hypothesis function indexed by θ.

→ Write the linear hypothesis function.

def h(x, theta):
    ...
Now that we have our linear regression model, we need to define a cost function to train it, i.e measure how well the model performs and fits the data.
One of the most commonly used function is the Root Mean Squared Error (RMSE). As it is a cost function, we will need to optimize it and find the value
of theta which minimizes it.

Since the sqrt function is monotonous and increasing, we can minimize the square of RMSE, the Mean Square Error (MSE) and it will lead to the same result.

         m
MSE(X, hθ) =   1⁄m ∑ (θT·x(i) - y(i))2
         k=1

• X is a matrix which contains all the feature values. There is one row per instance.
• m is the number of instances.
• xi is the feature values vector of the ith instance
• yi is the label (desired value) of the ith instance.

→ Write the Mean Squared Error function between the predicted values and the labels.

def mean_squared_error(y_predicted, y_label):
    ...
Now our goal is to minimize this MSE function.

Closed-Form Solution
To find the value of θ that minimizes the cost function, we can differentiate the MSE with respect to θ.
It directly gives us the correct θ in what we called the Normal Equation:

θ = (XT·X)-1·XT·y

(NB: This requires XTX to be inversible).

→ Write a class LeastSquareRegression to calculate the θ feature weights and make predictions.

hint Checkout Numpy's linear algebra module

class LeastSquaresRegression():
    def __init__(self,):
        self.theta_ = None  
        
    def fit(self, X, y):
        # Calculates theta that minimizes the MSE and updates self.theta_
    
        
    def predict(self, X):
        # Make predictions for data X, i.e output y = h(X) (See equation in Introduction)

Let's now use this class on data we are going to generate.
Here is some code to generate some random points.

import numpy as np
import matplotlib.pyplot as plt

X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)
→ Plot these points to get a feel of the distribution.

As you can see, these points are generated in a linear way with some Gaussian noise.
Before calculating our weights, we will account for the bias term (x0 = 1).

→ Write a function which adds one to each instance

def bias_column(X):
    ...

X_new = bias_column(x)

print(X[:5])
print(" ---- ")
print(X_new[:5])
You should see something similar to:

[[0.91340515]
 [0.14765626]
 [3.75646273]
 [2.23004972]
 [1.94209257]]
 ---- 
[[1.         0.91340515]
 [1.         0.14765626]
 [1.         3.75646273]
 [1.         2.23004972]
 [1.         1.94209257]]

→ Calculate the weights with the LeastSquaresRegression class

model = LeastSquaresRegression()
model.fit(X_new, y)

print(model.theta_)
→ Are the values consistent with the generating equation (i.e 10 and 2) ?

Let's see what our model predicts !

→ Use your model to predict values from X and plot the two set of points superimposed.

y_new = model.predict(X_new)

def my_plot(X, y, y_new):
    ...

my_plot(X, y, y_new)

You should see something similar to the pictures in the subject introduction.

→ What is the computational complexity of this method?
→ How does the training complexity compare to the predictions complexity?

Gradient Descent
Reminder about Gradient Descent
As you may have noticed, our MSE cost function is a convex function. This means that to find the minimum, a strategy based on a gradient descent
will always lead us to a global optimum. Remember that the gradient descent moves toward the direction of the steepest slope.

We will write a class to perform the gradient descent optimization.

class GradientDescentOptimizer():

    def __init__(self, f, fprime, start, learning_rate = 0.1):
        self.f_      = f                       # The function
        self.fprime_ = fprime                  # The gradient of f
        self.current_ = start                  # The current point being evaluated
        self.learning_rate_ = learning_rate    # Does this need a comment ?

        # Save history as attributes
        self.history_ = [start]
    
    def step(self):
        # Take a gradient descent step
        # 1. Compute the new value and update selt.current_
        # 2. Append the new value to history
        # Does not return anything

        
    def optimize(self, iterations = 100):
        # Use the gradient descent to get closer to the minimum:
        # For each iteration, take a gradient step

            
    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))

Let's use this optimizer with a simple function: f(x) = 3 + (x - (2  6)T)T · (x - (2  6)T). The input of f is a vector of size 2.

→ Write the f function

def f(x):
    ...
→ Write the fprime function

def fprime(x):
    ...
→ Use the the gradient descent optimizer to try to find the best theta value

grad = GradientDescentOptimizer(f, fprime, np.random.normal(size=(2,)), 0.1)
grad.optimize(10)
grad.print_result()
Don't hesitate to tweak the hyper parameters (the learning rate and the number of iterations).

→ Plot the function f in 3D

→ Plot the progression of the gradient by using the history variable inside the class

We should clearly see the gradient moving toward the cavity of the function f !

→ How does the learning rate and the number of iterations influence the result ?

Gradient Descent for Linear Regression
A gradient descent method is a good way to train our linear model. There exist many different implementation of the gradient descent.
For this example, we will implement a Batch Gradient Descent: we will use the whole data set.

To do so, we need to calculate the gradient of the cost function, the MSE function, with regards to the model weight vector θ
The gradien can be calculated thanks to the partial derivatives with regards to θj. The jth partial derivatives evaluates how much the function varies if θj changes a bit.
In the example of the hill at night, this is like testing multiple directions to find the steepest one.

The gradient is a vector whose ith value is the ith partial derivative.
After caculation, the gradient of the MSE with regards to θ is:

∇θMSE(θ) = 2⁄m XT·(X·θ - y)

Once we know the expression of the gradient vector, the strategy to update θ is the same as the one we have seen before:

θ(n + 1) = θ(n) - α∇θMSE(θ)

→ Implement the batch gradient descent


theta_start = np.random.randn(2,1)
def gradient_descent(X, m, y, theta_start, iterations = 100, learning_rate = 0.1):
    ...

theta = gradient_descent(X_new, 100, y, theta_start)
print(theta)

→ Is the result similar to the one found with the Normal Equation?

→ Plot for the first 10-20 steps the line represented by theta, with a learning rate of 0.01, 0.1, and 0.7.

y_predicted = h(X, theta);

plt.plot(X, y, "b.")
plt.plot(X, y_predicted, "r-")
plt.grid()

→ How does the learning rate influence the results?

→ How to tune hyperparameters and choose a good learning rate?

→ What about the number of iterations? What is the Convergence Rate?
