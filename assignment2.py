"""
In this assignment you should find the intersection points for two functions.
"""
import math

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def find_root_bisection_method(self, a, b, maxerr, func):
        # a,b being 2 initial guesses bracketing the root (different sign)
        delta = 2 * maxerr
        while not abs(b - a) < delta:
            z = (a + b) / 2
            if func(a) * func(z) < 0:
                b = z
            else:
                a = z
        return (a + b) / 2  # (a+b)/2 should be closer to the root than a or b statistically

    def check_bracketing(self, a, b, f):
        """
        To be used for Bisection method
        """
        if f(a) * f(b) < 0:
            return True
        else:
            return False

    def find_root_newton_raphson_bisec(self, a, b, maxerr, func,max_iter = 9):
        '''
        This function implements the Newton Raphson algorithm
        and in failed case moves directly to bisection method
        return: A single root of func in the given range [a,b]
        '''
        ass = Assignment2()
        x0 = random.uniform(a, b)
        find_der_in_point = lambda f, x, h=0.0001: (f(x + h) - f(x)) / h
        for n in range(0, max_iter):
            try:
                f_x0 = func(x0)
                der_x0 = find_der_in_point(func, x0)
            except ValueError:
                if ass.check_bracketing(a, b, func):
                    return ass.find_root_bisection_method(a, b, maxerr, func)
                return None
            if abs(f_x0) < maxerr:
                return x0
            if der_x0 == 0:
                if ass.check_bracketing(a, b, func):
                    return ass.find_root_bisection_method(a, b, maxerr, func)
                return None
            x0 = x0 - f_x0 / der_x0
        if ass.check_bracketing(a, b, func):
            return ass.find_root_bisection_method(a, b, maxerr, func)
        return None

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points. 

        Approach: In order to improve running times and since I wish to find all the roots,
        I'll only visit a section if there is a root in it (via intermediate theorem)

        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        X = []  # list of roots found during the run
        intersect_func = lambda x: 100000 * (f1(x) - f2(x))
        # creates the intersect function out of f1 and f2 - both parameters in main function
        ass = Assignment2()

        # I'll create a loop scanning the domain [a,b] from side to side
        # every section of the domain will be checked using the root finding methods by hierarchy
        h = 1/50  # Create section small enough
        left = a
        right = a + h

        while right <= b:
            # Are the limits of the section the roots? if so -> finish loop on section
            if abs(intersect_func(left)) < maxerr:
                X.append(left)

            elif abs(intersect_func(right)) < maxerr:
                X.append(right)

            # Check for bracketing, Else, leave section
            else:
                root = ass.find_root_newton_raphson_bisec(left, right, maxerr, intersect_func)
                if root is not None and left <= root <= right:
                    X.append(root)
                    # If found root -> take a smaller step forward
                    left = root + h
                    right = left + h
                    continue
            left = right
            right += h

        return X


##############################################################################################


import unittest
from sampleFunctions import *


# from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    # def test_sqr(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = np.poly1d([-1, 0, 1])
    #     f2 = np.poly1d([1, 0, -1])
    #
    #     X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
    #     print(X)
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #
    # def test_poly(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1, f2 = randomIntersectingPolynomials(10)
    #
    #     X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
    #     print(X)
    #     for x in X:
    #         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_Area_between(self):
        ass2 = Assignment2()
        f1 = lambda x: math.sin(math.log(x))
        f2 = lambda x: pow(x, 2) - 3 * x + 2
        X = ass2.intersections(f1, f2, 1, 3, maxerr=0.001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
