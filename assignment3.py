"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
import math
##################################################################


def find_root_bisection_method(a, b, maxerr, func):
    # a,b being 2 initial guesses bracketing the root (different sign)
    delta = 2 * maxerr
    while not abs(b - a) < delta:
        z = (a + b) / 2
        if func(a) * func(z) < 0:
            b = z
        else:
            a = z
    return (a + b) / 2  # (a+b)/2 should be closer to the root than a or b statistically


def check_bracketing(a, b, f):
    if f(a) * f(b) < 0:
        return True
    else:
        return False


def find_root_newton_raphson_bisec(a, b, maxerr, func, max_iter=9):
    '''
    This function implements the Newton Raphson algorithm
    and in failed case moves directly to bisection method
    return: A single root of func in the given range [a,b]
    '''
    # x0 = random.uniform(a, b)
    x0 = (a + b) / 2  # initial point as random -> middle of section

    der = lambda f, x, h=1e-8: ((f(x + h) - f(x)) / h)

    for i in range(max_iter):  # Forloop to prevent recalculation every iteration
        f_x0 = func(x0)
        der_x0 = der(func, x0)
        if der_x0 == 0:  # Zero derivative, No solution found
            if check_bracketing(a, b, func):
                return find_root_bisection_method(a, b, maxerr, func)  # Failed? try bisection
            return None
        x0 = x0 - f_x0 / der_x0
        if (x0 < a) or (x0 > b):  # X is out of range and might cause Valuerror, exit function
            break

        if abs(f_x0) < maxerr:  # Found solution.
            return x0
    # Exceeded maximum iterations, No solution found. Try bisection.
    if check_bracketing(a, b, func):
        return find_root_bisection_method(a, b, maxerr, func)  # Failed? try bisection
    return None


def intersections(f1: callable, f2: callable, a: float, b: float, maxerr):
    X = []  # list of roots found during the run
    intersect_func = lambda x: 100000 * (f1(x) - f2(x))
    # creates the intersect function out of f1 and f2 - both parameters in main function

    # I'll create a loop scanning the domain [a,b] from side to side
    # every section of the domain will be checked using the root finding methods by hierarchy
    h = abs(b-a)/500  # Create section small enough
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
            root = find_root_newton_raphson_bisec(left, right, maxerr, intersect_func)
            if root is not None:
                X.append(root)
                # If found root -> take a smaller step forward
                left = root + h
                right = left + h
                continue
        left = right
        right += h

    return X

##################################################################
# Recursive generation of the Legendre polynomial of order n
def Legendre(n, x):
    x = np.array(x)
    if n == 0:
        return x * 0 + 1.0
    elif n == 1:
        return x
    else:
        return ((2.0 * n - 1.0) * x * Legendre(n - 1, x) - (n - 1) * Legendre(n - 2, x)) / n


##################################################################
# Derivative of the Legendre polynomials
def DLegendre(n, x):
    x = np.array(x)
    if n == 0:
        return x * 0
    elif n == 1:
        return x * 0 + 1.0
    else:
        return (n / (x ** 2 - 1.0)) * (x * Legendre(n, x) - Legendre(n - 1, x))


##################################################################
# Roots of the polynomial obtained using Newton-Raphson method
def LegendreRoots(polyorder, tolerance=1e-20):
    if polyorder < 2:
        err = 1  # bad polyorder no roots can be found
    else:
        roots = []
        # The polynomials are alternately even and odd functions. So we evaluate only half the number of roots.
        for i in range(1, int(polyorder / 2) + 1):
            x = np.cos(np.pi * (i - 0.25) / (polyorder + 0.5))
            error = 10 * tolerance
            iters = 0
            while (error > tolerance) and (iters < 1000):
                dx = -Legendre(polyorder, x) / DLegendre(polyorder, x)
                x = x + dx
                iters = iters + 1
                error = abs(dx)
            roots.append(x)
        # Use symmetry to get the other roots
        roots = np.array(roots)
        if polyorder % 2 == 0:
            roots = np.concatenate((-1.0 * roots, roots[::-1]))
        else:
            roots = np.concatenate((-1.0 * roots, [0.0], roots[::-1]))
        err = 0  # successfully determined roots
    return [roots, err]


##################################################################
# Weight coefficients
def GaussLegendreWeights(polyorder):
    W = []
    [xis, err] = LegendreRoots(polyorder)
    if err == 0:
        W = 2.0 / ((1.0 - xis ** 2) * (DLegendre(polyorder, xis) ** 2))
        err = 0
    else:
        err = 1  # could not determine roots - so no weights
    return [W, xis, err]


##################################################################
# The integral value
# func 		: the integrand
# a, b 		: lower and upper limits of the integral
# polyorder 	: order of the Legendre polynomial to be used (number of points)
#
def GaussLegendreQuadrature(func, polyorder, a, b):
    [Ws, xs, err] = GaussLegendreWeights(polyorder)
    result = 0
    for x, w in zip(xs, Ws):
        result += w * func((b - a) * 0.5 * x + (b + a) * 0.5)
    return (b - a) * 0.5 * result
    # return (b - a) * 0.5 * np.sum(Ws * func((b - a) * 0.5 * xs + (b + a) * 0.5))


def integrate_Gauss_Quadrature(f, a, b, n):
    # let n be the maximal number of points that can be sampled from f
    # let's use Gausses method for splines of the function, using 2 to 3 points at the time
    S = 0
    points_per_section = 2
    if n % 2 != 0:
        n -= 1
    num_sections = n // points_per_section
    for i in range(num_sections):
        # divide the range into sections
        start = a + i * (b - a) / num_sections
        end = a + (i + 1) * (b - a) / num_sections
        S += GaussLegendreQuadrature(f, points_per_section, start, end)

    # handle the remaining points
    remaining_points = n % points_per_section
    if remaining_points > 0:
        start = b - (b - a) / num_sections
        S += GaussLegendreQuadrature(f, remaining_points, start, b)

    return np.float32(S)


def integrateSimpson(f: callable, a: float, b: float, n: int) -> np.float32:
    if n == 1:
        return np.float32(0)

    if n % 2 == 0:
        n -= 1

    #  Divide into sections
    h = (b - a) / (n - 1)
    xs = np.linspace(a, b, n)  # n points make n-1 sub-intervals
    ys = np.array([f(x) for x in xs])

    # Interpolation using Simpson's rule of 2nd degree - Closed
    S = h / 3 * np.sum(ys[0:-1:2] + 4 * ys[1::2] + ys[2::2])
    return np.float32(S)


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        return integrate_Gauss_Quadrature(f, a, b, n)



    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        f = lambda x: f1(x) - f2(x)
        maxerr = 0.001
        points = np.array(intersections(f1, f2, 1, 100, maxerr), dtype=np.float32)
        points = sorted(points)  # Make sure the points are sorted
        if len(points) > 1:
            total = 0
            for i in range(len(points) - 1):
                if not (points[i+1] - points[i]) < 2 * maxerr:  # If the points are not "duplicated"
                    total += abs(integrateSimpson(f, points[i], points[i + 1], 100))
                else:
                    continue
            return np.float32(total)
        return np.nan

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = lambda x: x ** 2 - 1
        r = ass3.integrate(f1, 0, 4, 100)
        true_result = 52 / 3
        print('intergrate =', r)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))
        # f1 = strong_oscilations()
        # r = ass3.integrate(f1, 0.09, 10, 20)
        # true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))
        f1 = lambda x: np.sin(x ** 2)
        r = ass3.integrate(f1, 100, 1.5, 200)
        print('intergrate =', r)

        self.assertEquals(r.dtype, np.float32)

    def test_Area_between(self):
        ass3 = Assignment3()
        f1 = lambda x: (x - 10) ** 2
        f2 = lambda x: 4
        S = ass3.areabetween(f1, f2)
        print('area =', S)

        f1 = lambda x: math.sin(math.log(x))
        f2 = lambda x: pow(x, 2) - 3 * x + 2
        S = ass3.areabetween(f1, f2)
        print('area =', S)
        print('Error =', S - 0.731257)


if __name__ == "__main__":
    unittest.main()

