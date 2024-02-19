import numpy as np
from ..helper_funcs.helper_funcs import comb
import scipy as sp


class Coalescence:
    """
    A class that represents a Coalescence matrix and provides methods to produce the corresponding
    Fst matrix and migration matrix.
    The production of the Fst matrix is according to Slatkin's 1991 paper.
    The production of the migration matrix is according to Wilkinson-Herbot's 2003 paper, and Xiran Liu's 2019 paper.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize a coalescence times matrix object
        :param matrix: input Coalescence time matrix
        """
        self.matrix = matrix
        self.shape = matrix.shape[0]

    def produce_fst(self) -> np.ndarray:
        """
        produces and returns the corresponding Fst matrix
        :return: The corresponding Fst matrix
        """
        F_mat = np.zeros((self.shape, self.shape))
        for i in range(self.shape):
            for j in range(i + 1, self.shape):
                t_S = (self.matrix[i, i] + self.matrix[j, j]) / 2
                t_T = (self.matrix[i, j] + t_S) / 2
                if np.isinf(t_T):
                    F_i_j = 1
                else:
                    F_i_j = (t_T - t_S) / t_T
                F_mat[i, j], F_mat[j, i] = F_i_j, F_i_j
        return F_mat

    def produce_migration(self, bounds=(0, 2)) -> tuple:
        """
        produce and return the migration matrix induced by the coefficient matrix A (which is induced by T).
        :param bounds: bounds for each individual variable. default is 0 < x < 2. bounds should be given as a tuple
                       of 2 arrays of size n**2-n (where n is the number of populations). first array represents lower
                       bounds, second
                       array represents upper bounds. if a tuple with 2 scalars is given instead,
                       they will be the bounds for each variable.
        :return: (Migration matrix corresponding to object's Coalescence matrix, output of scipy.optimize.lsq_linear).
        """
        n = self.shape
        M = np.zeros((n, n))
        A = self.produce_coefficient_mat()
        b = self.produce_solution_vector()
        ls_sol = sp.optimize.lsq_linear(A, b, bounds=(bounds[0], bounds[1]), max_iter=1000)
        x = ls_sol.x
        # norm = 0.5 * np.linalg.norm(A @ x - b, ord=2) ** 2
        for i in range(n):
            start_ind = i * (n - 1)
            M[i, 0:i] = x[start_ind:start_ind + i]
            M[i, i + 1:n] = x[start_ind + i:start_ind + n - 1]
        return M, ls_sol

    def produce_coefficient_mat(self) -> np.ndarray:
        """
        produces and returns the corresponding coefficient matrix, taking into consideration the assumption of
        conservative migration.
         :return: The corresponding coefficient matrix (A)
         """
        n = self.shape
        n_rows = 2 * n + comb(n, 2)  # number of equations
        n_cols = n ** 2 - n  # number of unknowns
        A = np.zeros((n_rows, n_cols))
        for i in range(n):
            s_i, e_i, v_i = self.s(i), self.e(i), self.v(i)
            A[v_i, s_i:e_i + 1] = 1
            counter = 0
            for j in range(n):
                if j != i:
                    A[i, s_i + counter] = self.matrix[i, i] - self.matrix[i, j]
                    counter += 1
                if j < i:
                    A[v_i, j * (n - 1) + i - 1] = -1
                if j > i:
                    A[v_i, j * (n - 1) + i] = -1

            if i != 0:
                for j in range(i):
                    counter_i, counter_j = 0, 0
                    w_i_j = self.w(i, j)
                    s_j = self.s(j)
                    for k in range(n):
                        if k != j:
                            A[w_i_j, s_j + counter_j] = self.matrix[i, j] - self.matrix[i, k]
                            counter_j += 1
                        if k != i:
                            A[w_i_j, s_i + counter_i] = self.matrix[i, j] - self.matrix[j, k]
                            counter_i += 1
        return A

    def s(self, i):
        return i * (self.shape - 1)

    def e(self, i):
        return (i + 1) * (self.shape - 1) - 1

    def w(self, i, j):
        return int(self.shape + ((i * (i - 1)) / 2) + j)

    def v(self, i):
        return self.shape + comb(self.shape, 2) + i

    def produce_solution_vector(self) -> np.ndarray:
        """
        produce the solution vector(b), according to Wilkinson-Herbot's equations.
        :return: solution vector b
        """
        n = self.shape
        nC2 = comb(n, 2)
        b = np.zeros((2 * n + nC2))
        b[0:n] = 1 - self.matrix.diagonal()
        b[n: n + nC2] = 2
        return b
