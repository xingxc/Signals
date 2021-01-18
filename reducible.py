# %% O-notation learn
import numpy as np
import time


class grid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.grid = self.grid_generation()

    def grid_generation(self):
        output = np.zeros((self.x, self.y))
        num = 1
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = num
                num += 1
        return output

    def grid_path_count(self, row=None, col=None):
        if row == None:
            row = self.grid.shape[0]
        if col == None:
            col = self.grid.shape[1]
        if row == 1 or col == 1:
            return 1
        else:
            return self.grid_path_count(row, col-1) + self.grid_path_count(col-1, row)


def partition(n, m, count):
    if n == 0:
        return 1
    if m == 0 or n < 0:
        return 0
    else:
        # print(count)
        count += 1
        # logger.append([partition(n-m, k), partition(n, m-1)])
        return partition(n, m-1, count) + partition(n-m, m, count)


def solution_bob(n):
    start_time = time.time()
    a = np.arange(0, n+1, 1)
    b = np.arange(0, n+1, 1)
    c = np.arange(0, n+1, 1)
    iterations = 0
    output = []

    for a_temp in a:
        for b_temp in b:
            for c_temp in c:
                iterations += 1
                if (a_temp + b_temp + c_temp == n):
                    output.append([a_temp, b_temp, c_temp])

    output = np.array(output)
    end_time = time.time()

    return iterations, end_time-start_time, output


def solution_alice(n):
    start_time = time.time()
    a = np.arange(0, n+1, 1)
    b = np.arange(0, n+1, 1)

    iterations = 0
    a_b = []
    output = []

    for a_temp in a:
        for b_temp in b:
            iterations += 1
            a_b.append(a_temp + b_temp)
            if (n - (a_temp + b_temp) >= 0):
                c_temp = n - (a_temp + b_temp)
                output.append([a_temp, b_temp, c_temp])

    end_time = time.time()
    return iterations, end_time-start_time, output
