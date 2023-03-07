"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import math
import time
import random
from functionUtils import AbstractShape


def BoxcarAVGpoint(Points, k):
    '''
    I know the data will be noisy, so I'll use every k number of points to create an average point
    '''
    # Let n be an odd small number, 3 or 5 for example, so it won't damage the shape
    # I'll concat the first n-1 items again to the points array so the calculation can be done
    # for every point as a center point

    n = len(Points)
    Points = [p for p in Points]  # Reshape it into list
    Points = Points + Points[:k]
    Points = np.array(Points)  # Reshape it into np.array
    FinalPoints = []
    for i in range(n):
        tmp = Points[i:i+k]
        x_mean = np.mean(tmp[:, 0])
        y_mean = np.mean(tmp[:, 1])
        FinalPoints.append(np.array([x_mean, y_mean]))
    return FinalPoints


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, points):
        #super(MyShape, self).__init__()
        self.points = points
        self.polygon_area = 0

        # Activate fit() to get the points on the shape contour
        # Activate area to calculate the area once
        # Works as a contractor

    def area(self):
        # ass5.area(self, contour: callable)
        # Returns the area of the shape
        # Uses the area of the shape as attribute of the shape
        if self.polygon_area == 0:
            P = self.points
            x = [point[0] for point in P]
            y = [point[1] for point in P]
            self.polygon_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            return self.polygon_area
        else:
            return self.polygon_area


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        n = int(1 / maxerr)
        P = contour(n)
        x = [point[0] for point in P]
        y = [point[1] for point in P]
        # Use the shoelace method to return the area
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """
        start_time = time.time()
        max_n = 2000 * maxtime  # creating a maximum amount of points, max_n <= 50000 * maxtime
        pts = []
        counter = 0
        while counter < max_n:  # Collecting points until reaching 65% of maxtime
            point = sample()
            pts.append(list(point))
            counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.95 * maxtime:
                break

        # print(f'Your model uses {n} points - number of points pulled')
        # subtracting the mean of all the x and y coordinates of the points (computed along axis 0)
        # from each x and y coordinate respectively, effectively centering the points around the origin.
        pts = np.array(pts)
        pts = (pts - pts.mean(0))
        # Sort the points based on their angle with the X axis (using complex number form)
        pts = pts[np.angle((pts[:, 0] + 1j * pts[:, 1])).argsort()]
        # Return the points as attribute of the shape
        # Finalized points will be used to calculate the shape's area
        # number of points for averaging will be 8 for small n or 500 sections for large n
        factor = max(8, int(len(pts)/500))
        final_points = BoxcarAVGpoint(pts, factor)
        result = MyShape(final_points)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        print("Test 1")
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        print(shape)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_my_fit_shape(self):
        print("Test 2")
        ass5 = Assignment5()
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=00.1)
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        print(shape)
        print(f'area: {shape.area()}')

    def test_delay(self):
        print("Test 6 - delay (complete test)")
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        r_circ = Circle(cx=1, cy=1, radius=1, noise=0.0)

        def sample():
            time.sleep(0.01)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        test_time = 5
        shape = ass5.fit_shape(sample=sample, maxtime=test_time)
        T = time.time() - T
        print(f'Fitting the shape took you {T} seconds')
        a = shape.area()
        a_computed = ass5.area(contour=r_circ.contour, maxerr=0.1)
        print(f'Area by points: {a}')
        print(f'Area by contour: {a_computed}')
        print(f'Real area: {np.pi}')
        print(f'Area difference (contour): {abs((np.pi - a_computed))}')
        print(f'Area difference (points): {abs((np.pi - a))}')
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLess(abs(a_computed - np.pi), 0.01)
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, test_time)

    def test_circle_area(self):
        print("Test 3")
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(shape)
        print(f'area: {a}')
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        print("Test 4")
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(shape)
        print(f'area: {a}')
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        print("Test 5")
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass5 = Assignment5()
        T = time.time()
        a_computed = ass5.area(contour=circ.contour, maxerr=0.1)
        print(circ)
        T = time.time() - T
        a_true = circ.area()

        print(f'True area = {a_true}')
        print(f'Calc area = {a_computed}')
        print(f'Relative error: {abs((a_true - a_computed)/a_true)}')
        self.assertLess(abs((a_true - a_computed)/a_true), 0.1)

if __name__ == "__main__":
    unittest.main()
