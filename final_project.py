import numpy as np
import matplotlib.pyplot as plt

'''
   The Sphere, Rosenbrock, and McCormick Test Functions for Optimization
'''

# Sphere Function, n = 2
def sphere(x1, x2):
    return x1 ** 2 + x2 ** 2

# gradient of Sphere Function
def sphere_grad(x1, x2):
    return 2 * x1 + 2 * x2

# Rosenbrock Function, n = 2
def rosenbrock(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

# gradient of Rosenbrock Function
def rosenbrock_grad(x1, x2):
    return (400 * x1 ** 3) - (400 * x1 * x2) + (2 * x1) - 2 - (200 * x1 ** 2) + (200 * x2)

# McCormick Function
def mcCormick(x1, x2):
    return np.sin(x1 + x2) + ((x1 - x2) ** 2) - (1.5 * x1) + (2.5 * x2) + 1

# gradient of McCormick Function
def mcCormick_grad(x1, x2):
    return np.cos(x1 + x2) + (2 * (x1 - x2)) - 1.5 + np.cos(x1 + x2) + 2.5 - (2 * (x1 - x2))

# Gradient Descent Function
def gradient_descent(x1, x2, gradient, learning_rate, max_iter, tol = 0.0000001, momentum = 0):
    steps = [x1, x2]
    diff = 0
    for _ in range(max_iter):
        diff = learning_rate * gradient(x1, x2) + momentum * diff
        if np.abs(diff) < tol:
            break

        x1 = x1 - diff
        x2 = x2 - diff

        steps.append([x1, x2])
    
    return steps

''' Gradient Descent of the Test Functions without Momentum'''

# Gradient descent of the sphere function
result1 = gradient_descent(100, 100, sphere_grad, 0.1, 15)

# Gradient descent of the Rosenbrock function
result2 = gradient_descent(5, 5, rosenbrock_grad, .0002, 3)

# Gradient descent of the McCormick function
result3 = gradient_descent(4, 4, mcCormick_grad, 0.9, 3)


'''Gradient Descent of the Test Functions with Momentum'''

# Gradient descent of the sphere function
result4 = gradient_descent(100, 100, sphere_grad, 0.1, 10, momentum=0.1)

# Gradient descent of the Rosenbrock function
result5 = gradient_descent(5, 5, rosenbrock_grad, .0001, 2, momentum=0.8)

# Gradient descent of the McCormick function
result6 = gradient_descent(4, 4, mcCormick_grad, .9, 2, momentum=1)

'''Sphere Figure'''

x1 = np.linspace(-100, 100, 100)
x2 = np.linspace(-100, 100, 100)

a=np.array((1,2))
a1,a2=np.meshgrid(a,a)
X1, X2 = np.meshgrid(x1, x2)
Y = np.sqrt(np.square(X1) + np.square(X2))
cm = plt.cm.get_cmap('viridis')
plt.scatter(X1, X2, c=Y, cmap=cm)
levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
cp = plt.contour(X1, X2, Y, levels, colors='white', linestyles='solid')
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X1, X2, Y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()
plt.plot()
plt.title('Gradient Descent on the Sphere Function')
plt.show()

'''Rosenbrock Figure'''

x1 = np.linspace(-5, 5)
x2 = np.linspace(-20, 5)
a=np.array((1,2))
a1,a2=np.meshgrid(a,a)
X1, X2 = np.meshgrid(x1, x2)
Y = (100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2)
cm = plt.cm.get_cmap('viridis')
plt.scatter(X1, X2, c=Y, cmap=cm)
cp = plt.contour(X1, X2, Y, colors='white', linestyles='solid')
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X1, X2, Y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()
plt.plot()
plt.title('Gradient Descent on the Rosenbrock Function')
plt.show()

'''McCormick Figure'''

x1 = np.linspace(-3, 4, 100)
x2 = np.linspace(-3, 4, 100)

a=np.array((1,2))
a1,a2=np.meshgrid(a,a)
X1, X2 = np.meshgrid(x1, x2)
Y = np.sin(X1 + X2) + ((X1 - X2) ** 2) - (1.5 * X1) + (2.5 * X2) + 1
cm = plt.cm.get_cmap('viridis')
plt.scatter(X1, X2, c=Y, cmap=cm)
cp = plt.contour(X1, X2, Y, levels, colors='white', linestyles='solid')
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X1, X2, Y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()
plt.plot()
plt.title('Gradient Descent on the McCormick Function')
plt.show()