"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
import math


def lagrange_interpolation(points):
    # Returns a function f(x) using lagrange_interpolation on k points
    def P(x):
        total = 0
        n = len(points)
        for i in range(n):
            xi, yi = points[i]

            def g(i, n):
                total = 1
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj = points[j]
                    total *= (x - xj) / (xi - xj)
                return total

            total += yi * g(i, n)
        return total

    return P


def binary_search(keys, x):
    #  Array like [[xi,xj],[xii,xjj],...,]
    low = 0
    high = len(keys) - 1
    # Indexes for keys
    while low <= high:
        mid = (high + low) // 2
        # If x is greater, ignore left half
        if keys[mid][1] < x:
            low = mid + 1
        # If x is smaller, ignore right half
        elif keys[mid][0] > x:
            high = mid - 1
        # means x is present at mid
        else:
            return mid  # if xi <= x <= xj, return key = [xi,xj]


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        if n == 1:  # In case we can only use 1 point -> return a straight line
            f = lambda x: f((a + b) / 2)
            return f
        # Next we will create 2 arrays: X values and f(X) values
        # Split the range to n - points -> since every section is...
        # ...independent there is not much point for Chebieshev
        x_values = np.linspace(a, b, n)
        y_values = [f(x) for x in x_values]  # Get the f(x) values
        points = [[x_values[i], y_values[i]] for i in range(n)]  # Create n points in the form of (x,y)

        # split into group of n points (re-use x2...xn points from every group)
        def split_list(input_list, chunk_size, end=b):
            output = []
            start = 0
            diff = 2
            while start + chunk_size <= len(input_list):
                output.append(input_list[start:start + chunk_size])
                start += diff
            if output[-1][-1][0] < end:
                output.append(input_list[-chunk_size:])  # Last group of points
            return output

        # use to create groups of 7 points
        Points = split_list(points, 7)

        # Each function will be saved in a dictionary:
        functions_dict = {}
        # edge of range:
        functions_dict[points[0][0], points[2][0]] = lagrange_interpolation(Points[0])
        for Pi in Points:
            # The interpolation for section [xi,x_i + 6]
            functions_dict[Pi[2][0], Pi[4][0]] = lagrange_interpolation(Pi)

        # edge of range:
        coverage = list(functions_dict.keys())[-1][1]
        functions_dict[coverage, points[-1][0]] = lagrange_interpolation(Points[-1])

        # Create a function that returns an interpolation function -> for x returns f(x)
        keys = list(functions_dict.keys())

        def interpolatz_PLATZ(x):
            pos = binary_search(keys, x)
            y = functions_dict[keys[pos]](x)  # Extract y-value
            return y

        # This line will hold the interpolation function as a returned parameter f(x_given) = y_value
        return interpolatz_PLATZ


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 20
        #for i in tqdm(range(100)):
        a = np.random.randn(d)

        f = np.poly1d(a)

        ff = ass1.interpolate(f, -10, 10, 100)
        index = -10
        for i in range(20):
            xs = np.random.uniform(index, index + 1, 20)
            index += 1
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

                err = err / 20
                mean_err += err
            mean_err = mean_err / 100
            print(f'Error for range {index - 1, index} is {mean_err}')

        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
