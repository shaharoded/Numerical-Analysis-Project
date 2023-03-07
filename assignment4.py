"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import math

import pandas as pd
import numpy as np
import random
import time

def np_inverse(A):
    try:
        return np.linalg.inv(A)

    except:
            # Except all kind of errors to catch float devision or math domain as well
            # Matrix is singular and cannot be inverted normally
            # invert using Moore-Penrose pseudoinverse of a matrix algorithm
        return np.linalg.pinv(A)


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        matrix0 = np.array([[1.0]], dtype=np.float64)
        matrix1 = np.array([[-1, 1], [1, 0]], dtype=np.float64)
        matrix2 = np.array([[1.0, -2.0, 1.0], [-2.0, 2.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        matrix3 = np.array([[-1.0, 3.0, -3.0, 1.0], [3.0, -6.0, 3.0, 0.0], [-3.0, 3.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                           dtype=np.float64)
        matrix4 = np.array([[1.0, -4.0, 6.0, -4.0, 1.0], [-4.0, 12.0, -12.0, 4.0, 0.0], [6.0, -12.0, 6.0, 0.0, 0.0],
                            [-4.0, 4.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix5 = np.array([[-1.0, 5.0, -10.0, 10.0, -5.0, 1.0], [5.0, -20.0, 30.0, -20.0, 5.0, 0.0],
                            [-10.0, 30.0, -30.0, 10.0, 0.0, 0.0], [10.0, -20.0, 10.0, 0.0, 0.0, 0.0],
                            [-5.0, 5.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix6 = np.array([[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0], [-6.0, 30.0, -60.0, 60.0, -30.0, 6.0, 0.0],
                            [15.0, -60.0, 90.0, -60.0, 15.0, 0.0, 0.0], [-20.0, 60.0, -60.0, 20.0, 0.0, 0.0, 0.0],
                            [15.0, -30.0, 15.0, 0.0, 0.0, 0.0, 0.0], [-6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix7 = np.array(
            [[-1.0, 7.0, -21.0, 35.0, -35.0, 21.0, -7.0, 1.0], [7.0, -42.0, 105.0, -140.0, 105.0, -42.0, 7.0, 0.0],
             [-21.0, 105.0, -210.0, 210.0, -105.0, 21.0, 0.0, 0.0], [35.0, -140.0, 210.0, -140.0, 35.0, 0.0, 0.0, 0.0],
             [-35.0, 105.0, -105.0, 35.0, 0.0, 0.0, 0.0, 0.0], [21.0, -42.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [-7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix8 = np.array([[1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0],
                            [-8.0, 56.0, -168.0, 280.0, -280.0, 168.0, -56.0, 8.0, 0.0],
                            [28.0, -168.0, 420.0, -560.0, 420.0, -168.0, 28.0, 0.0, 0.0],
                            [-56.0, 280.0, -560.0, 560.0, -280.0, 56.0, 0.0, 0.0, 0.0],
                            [70.0, -280.0, 420.0, -280.0, 70.0, 0.0, 0.0, 0.0, 0.0],
                            [-56.0, 168.0, -168.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [28.0, -56.0, 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [-8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix9 = np.array([[-1.0, 9.0, -36.0, 84.0, -126.0, 126.0, -84.0, 36.0, -9.0, 1.0],
                            [9.0, -72.0, 252.0, -504.0, 630.0, -504.0, 252.0, -72.0, 9.0, 0.0],
                            [-36.0, 252.0, -756.0, 1260.0, -1260.0, 756.0, -252.0, 36.0, 0.0, 0.0],
                            [84.0, -504.0, 1260.0, -1680.0, 1260.0, -504.0, 84.0, 0.0, 0.0, 0.0],
                            [-126.0, 630.0, -1260.0, 1260.0, -630.0, 126.0, 0.0, 0.0, 0.0, 0.0],
                            [126.0, -504.0, 756.0, -504.0, 126.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [-84.0, 252.0, -252.0, 84.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [36.0, -72.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [-9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix10 = np.array([[1.0, -10.0, 45.0, -120.0, 210.0, -252.0, 210.0, -120.0, 45.0, -10.0, 1.0],
                             [-10.0, 90.0, -360.0, 840.0, -1260.0, 1260.0, -840.0, 360.0, -90.0, 10.0, 0.0],
                             [45.0, -360.0, 1260.0, -2520.0, 3150.0, -2520.0, 1260.0, -360.0, 45.0, 0.0, 0.0],
                             [-120.0, 840.0, -2520.0, 4200.0, -4200.0, 2520.0, -840.0, 120.0, 0.0, 0.0, 0.0],
                             [210.0, -1260.0, 3150.0, -4200.0, 3150.0, -1260.0, 210.0, 0.0, 0.0, 0.0, 0.0],
                             [-252.0, 1260.0, -2520.0, 2520.0, -1260.0, 252.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [210.0, -840.0, 1260.0, -840.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-120.0, 360.0, -360.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [45.0, -90.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix11 = np.array([[-1.0, 11.0, -55.0, 165.0, -330.0, 462.0, -462.0, 330.0, -165.0, 55.0, -11.0, 1.0],
                             [11.0, -110.0, 495.0, -1320.0, 2310.0, -2772.0, 2310.0, -1320.0, 495.0, -110.0, 11.0, 0.0],
                             [-55.0, 495.0, -1980.0, 4620.0, -6930.0, 6930.0, -4620.0, 1980.0, -495.0, 55.0, 0.0, 0.0],
                             [165.0, -1320.0, 4620.0, -9240.0, 11550.0, -9240.0, 4620.0, -1320.0, 165.0, 0.0, 0.0, 0.0],
                             [-330.0, 2310.0, -6930.0, 11550.0, -11550.0, 6930.0, -2310.0, 330.0, 0.0, 0.0, 0.0, 0.0],
                             [462.0, -2772.0, 6930.0, -9240.0, 6930.0, -2772.0, 462.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-462.0, 2310.0, -4620.0, 4620.0, -2310.0, 462.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [330.0, -1320.0, 1980.0, -1320.0, 330.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-165.0, 495.0, -495.0, 165.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [55.0, -110.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-11.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        matrix12 = np.array([[1.0, -12.0, 66.0, -220.0, 495.0, -792.0, 924.0, -792.0, 495.0, -220.0, 66.0, -12.0, 1.0],
                             [-12.0, 132.0, -660.0, 1980.0, -3960.0, 5544.0, -5544.0, 3960.0, -1980.0, 660.0, -132.0,
                              12.0, 0.0],
                             [66.0, -660.0, 2970.0, -7920.0, 13860.0, -16632.0, 13860.0, -7920.0, 2970.0, -660.0, 66.0,
                              0.0, 0.0],
                             [-220.0, 1980.0, -7920.0, 18480.0, -27720.0, 27720.0, -18480.0, 7920.0, -1980.0, 220.0,
                              0.0, 0.0, 0.0],
                             [495.0, -3960.0, 13860.0, -27720.0, 34650.0, -27720.0, 13860.0, -3960.0, 495.0, 0.0, 0.0,
                              0.0, 0.0],
                             [-792.0, 5544.0, -16632.0, 27720.0, -27720.0, 16632.0, -5544.0, 792.0, 0.0, 0.0, 0.0, 0.0,
                              0.0],
                             [924.0, -5544.0, 13860.0, -18480.0, 13860.0, -5544.0, 924.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-792.0, 3960.0, -7920.0, 7920.0, -3960.0, 792.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [495.0, -1980.0, 2970.0, -1980.0, 495.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-220.0, 660.0, -660.0, 220.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [66.0, -132.0, 66.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [-12.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.Bezier_Matrix = {0: matrix0, 1: matrix1, 2: matrix2, 3: matrix3, 4: matrix4, 5: matrix5, 6: matrix6, 7: matrix7, 8: matrix8,
                         9: matrix9, 10: matrix10, 11: matrix11, 12: matrix12}


    def bezier_curve_iterative(self, points, degree):
        """
        points : array of shape (n,2) containing n control points of bezier curve.
        degree : degree of the bezier curve
        """

        def f(t):
            current_points = points
            for _ in range(degree):
                next_points = np.array([(1 - t) * current_points[i] + t * current_points[i + 1] for i in
                                        range(current_points.shape[0] - 1)])
                current_points = next_points
            return current_points[0]

        return f

    def OptimizedPoints(self, x_values, Points, d):  # x_values of my sampled points
        """
        Creation of matrix T: Let T be a matrix of 4 columns of the sort [...t^3, t^2, t, 1] and n rows so that each
        column holds the t values from the n points. each point will be given t value in [0,1] based on the relative
        distance it's x value has on the range [a,b] like: 0 <= ti = abs(xi-a)/(b-a) <= 1
        """
        # returns the |d| ideal points for the model
        M = self.Bezier_Matrix[d]
        M_inv = np_inverse(M)
        dx = x_values[-1] - x_values[0]  # My range
        a = x_values[0]
        f = lambda x: ((x - a) / dx)  # Return value is the relative t for xi
        t_values = [f(x) for x in x_values]
        f = lambda t: np.flip(np.power(t, range(d + 1), dtype=np.float64))  # f(t) -> create t array from every x
        T = [f(t) for t in t_values]

        # Let's multiply the matrices and create the ugliest one liner ever
        # create |d+1| optimate points for Bezier
        C = np.dot(np.dot(np.dot(M_inv, np_inverse(np.dot(np.transpose(T), T))), np.transpose(T)), Points)
        return C

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        d = 10 if d > 10 else d  # Model is limited to 10th degree polynomials for accuracy reasons in the Grader
        # Create 2 arrays: X values and f(X) values -> samples from the original functions
        # let n be the number of samples taken. No n limit, only time limit
        n = math.floor(11 * maxtime)
        x_values = np.linspace(a, b, n)  # Split the range to sorted n - points
        y_values = [f(x) for x in x_values]  # Get the f(x) values
        P = np.array(list(zip(x_values, y_values)))  # Create n points in the form of (x,y)

        C = self.OptimizedPoints(x_values, P, d)  # Insert changing d value
        x0 = C[0][0]
        xn = C[-1][0]

        # In case small d creates division by 0, return a line
        if xn-x0 == 0:
            return lambda x: C[0][1]

        fitted_func = self.bezier_curve_iterative(C, d)
        # Return a Bezier function given the points array (d+1 points) C as calculated, f(t)

        def Model_Shtraub(x):
            # need to translate f(t) to f(x)
            t = (x - x0) / (xn - x0)  # Get the relevant t from the x value
            y = fitted_func(t)[1]  # Returned object is a point thus extract y-value
            return y

        return Model_Shtraub


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        d = 0
        ff = ass4.fit(f=nf, a=0, b=1, d=d, maxtime=5)
        T = 5 - T
        print(f'You have {T} seconds left MotherFucker')
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(f'MSE for d = {d} is {mse}')

if __name__ == "__main__":
    unittest.main()
