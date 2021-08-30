"""
Author: Samuel Zhou
Date: 8/29/2021
Filename: all_inequality.py
Description: Contains class to solve a linear program framed as minimizing c^Tx subject to Ax >= b. 
"""

import numpy as np

class AllInequalitySolver():

    def zero(self, val):
        if abs(val) <= 0.000001:
            return 0
        else:
            return val

    def __init__(self, A, b, c, status):
        self.A, self.b, self.c = A, b, c
        self.m, self.n = self.A.shape
        self.status = status
        self.w = np.where(self.status == 1)[0]

        self.Aw = self.A[self.w]
        self.bw = self.b[self.w]
        self.mw = self.A.shape[0]

        self.x = np.linalg.solve(self.Aw, self.bw)
        self.r = A @ self.x - b
        self.r = np.vectorize(self.zero)(self.r)

        if np.any(self.r < 0):
            raise ValueError('The initial point is infeasible')

        if np.linalg.matrix_rank(self.Aw) != self.n:
            raise ValueError('The initial working set is singular')

    def max_step(self, Ap):
        t = 0
        maximum = np.inf

        for i in range(self.m):
            if Ap[i] < 0:
                sigma = self.r[i] / (-1 * Ap[i])

                if sigma <= maximum:
                    if sigma < maximum:
                        Apmin = 0
                    maximum = sigma
                    if Ap[i] < Apmin:
                        t = i
                        Apmin = Ap[i]

        return (maximum, t, maximum == np.inf)

    def solve(self):
        for i in range(50):
            y = np.linalg.solve(self.Aw.T, self.c)
            
            s, ys = 0, y[0]
            for i in range(1, self.n):
                if y[i] < ys:
                    s = i
                    ys = y[i]

            if ys >= 0:
                return self.x

            ws = self.w[s]
            es = np.zeros(self.n)
            es[s] = 1

            p = np.linalg.solve(self.Aw, es)
            Ap = self.A @ p
            Ap = np.vectorize(self.zero)(Ap)
            maximum, t, unbounded = self.max_step(Ap)
            if unbounded:
                print('The problem is unbounded')
                return False

            self.status[t] = 1
            self.status[ws] = 0
            self.w = np.where(self.status  == 1)[0]

            self.Aw = self.A[self.w]
            self.bw = self.b[self.w]
            self.x = self.x + maximum * p
            self.r = self.A @ self.x - self.b
            self.r = np.vectorize(self.zero)(self.r)


        return self.x

        
        

A = np.array([[28, 24, 25, 14, 31, 3, 15, 9, 1], [15, 15, 6, 2, 8, 0, 4, 10, 2], [6, 10, 2, 0, 15, 15, 0, 4, 120], [30, 20, 25, 15, 15, 0, 20, 30, 2], [20, 20, 20, 10, 8, 2, 15, 0, 2], [510, 370, 500, 370, 400, 220, 345, 110, 80], [34, 35, 42, 38, 42, 26, 27, 12, 20], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])
b = np.array([55, 100, 100, 100, 100, 2000, 350, 0, 0, 0, 0, 0, 0, 0, 0, 0])
c = np.array([1.84, 2.19, 1.84, 1.44, 2.29, 0.77, 1.29, 0.60, 0.72])
active = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])


solver = AllInequalitySolver(A, b, c, active)
x = solver.solve()
print(x @ c)
print(x)
