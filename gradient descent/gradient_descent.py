import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def get_data():
    """
	get 2D input data

	Returns:
	x1 -- first column of dataset
    x2 -- second column of dataset
	"""
    dataset = np.genfromtxt('planckNoisy30.csv', delimiter=",")

    x1 = dataset[:,0]
    x2 = dataset[:,1]

    return x1, x2


def plot_graph(x1, x2, y1, y2):
    """
	Draw plot of input dataset and it's linear approximation with
    gradient descent

	Arguments:
	x1 -- first column of dataset
    x2 -- second column of dataset
    y1 -- first column of linear approximation
    y2 -- second column of linear approximation
	"""

    plt.plot(x1, x2, '.', y1, y2, '-')
    plt.ylabel('output')
    plt.xlabel('input')
    plt.legend(['input data', 'linear approximation'])
    plt.show()


def compute_error(x1, x2, m, b):
    """
	compute the error between linear function and datapoints

	Arguments:
	x1 -- first column of dataset
    x2 -- second column of dataset
    m  -- gradient of the linear function
    b  -- y-intercept value

	Returns:
	average_error -- average error over all training examples
	"""

    desired_out = np.add(np.multiply(m, x1), b)
    actual_out = x2
    error = np.power(np.add(desired_out, np.negative(actual_out)), 2)
    average_error = np.sum(error)/x1.size

    return average_error

def compute_gradient(x1, x2, m_current, b_current, alpha=0.1):
    """
	compute on epoch correcting values for the linear function

	Arguments:
	x1 -- first column of dataset
    x2 -- second column of dataset
    m_current  -- current gradient of the linear function
    b_current  -- current y-intercept value of the linear function
    alpha -- learningrate

	Returns:
	m_new -- corrected, new gradient
    b_new -- corrected, new y-intercept value
	"""

    size = x1.size

    m_gradient = np.sum(np.multiply(np.multiply(2, x1), (np.add(np.add(np.multiply(m_current, x1), b_current), np.negative(x2))))) / size
    b_gradient = np.sum(np.multiply(2, (np.add(np.add(np.multiply(m_current, x1), b_current), np.negative(x2))))) / size

    m_new = m_current - alpha * m_gradient
    b_new = b_current - alpha * b_gradient

    return m_new, b_new


def run_gradient_descent(x1, x2, m, b, epochs=100, alpha=0.5):
    """
	run the full gradient descent

	Arguments:
	x1 -- first column of dataset
    x2 -- second column of dataset
    m  -- gradient of the linear function
    b  -- y-intercept value
    epochs -- number of epochs
    alpha -- learningrate

	Returns:
	average_error -- average error over all training examples
	"""

    m_current = m
    b_current = b

    for i in range(epochs):

        print(compute_error(x1, x2, m_current, b_current))
        m_current, b_current = compute_gradient(x1, x2, m_current, b_current, alpha=alpha)
        sleep(0.5)

    plot_graph(x1, x2, [-2, 2], [-2 * m_current+b_current, 2*m_current+b_current])


x1, x2 = get_data()
run_gradient_descent(x1, x2, 0, 0)
