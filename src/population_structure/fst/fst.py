import numpy as np
from scipy.optimize import minimize
from ..helper_funcs.helper_funcs import compute_coalescence, comb, constraint_generator, f_to_m, \
    cons_migration_constraint_generator


class Fst:
    """
    A class that represents a Fst matrix and provides methods to produce corresponding coalescence and migration
    matrices.
    The production of the coalescence matrix is according to Slatkin's 1991 paper.
    The production of the migration matrix is according to Wilkinson-Herbot's 2003 paper, and Xiran Liu's 2019 paper.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize Fst matrix object
        :param matrix: input Fst matrix
        """
        self.matrix = matrix
        self.shape = matrix.shape[0]

    def produce_coalescence(self, x0=None, constraint=False, bounds=(0, np.inf)) -> tuple:
        """
        generates a possible corresponding coalescence times matrix and returns it.
        Solves Slatkin's equations using a numerical solver.
        :param x0: initial guess for the variables, default is a random vector with bounds (0,2*n), where n is the
                   size of the matrix (number of populations).
        :param constraint: indicated whether the T matrix produced should be 'compliant'. default is False
        :param bounds: bounds for each variable T(i,j), default is (0, inf). bounds should be an array of tuples, each
                       is (min, max) pair of the corresponding variable. If bounds is a tuple of two scalars,
                       the same bounds are applied for each variable.
        :return: A tuple: (A possible corresponding Coalescence time matrix- T, details about
                          the solution of the numerical solver).
        """
        n, nc2 = self.shape, comb(self.shape, 2)
        if x0 is None:
            x0 = np.random.uniform(low=0, high=2 * n, size=(n + nc2,))
        T = np.zeros((n, n))
        f_values = self.matrix[np.triu_indices(n, 1)]
        # add constraints
        constraints = None
        if constraint:
            constraints = []
            row, col = 0, 1
            for i in range(nc2):
                constraint_1 = constraint_generator(i, nc2 + row)
                constraint_2 = constraint_generator(i, nc2 + col)
                col += 1
                constraints.append({"type": "ineq", "fun": constraint_1})
                constraints.append({"type": "ineq", "fun": constraint_2})
                if col == n:  # move to next row
                    row += 1
                    col = row + 1
        solution = minimize(compute_coalescence, x0=x0, args=(f_values, n), bounds=(n + nc2) * [(bounds[0], bounds[1])],
                            constraints=constraints)
        x = solution.x
        np.fill_diagonal(T, x[nc2:])
        row_indices, col_indices = np.triu_indices(n, 1)
        T[(row_indices, col_indices)] = x[0:nc2]
        T[(col_indices, row_indices)] = x[0:nc2]
        return T, solution

    def produce_migration(self, x0=None, bounds=(0, np.inf), conservative=True) -> tuple:
        """
        produces and returns the migration matrix induced by the Fst matrix, using a numerical solver.
        This is a direct approach where the migration matrix is produced directly from the Fst matrix, without the
        intermediate step of producing the coalescence matrix.
        :param x0: Starting point for the numerical solver. if None, a random vector with bounds (0 ,2n) is used.
        :param bounds: bounds for each unknown migration value. Should ba tuple of two scalars, default is (0, inf).
        :param conservative:  Indicates whether the migration matrix should be conservative. default is True. This
                              does not guarantee that the migration matrix will be conservative!
        :return: A tuple (matrix, solution).
                One possible corresponding migration matrix, according to W.H and Slatkin's equations, and details
                about the solution of the numerical solver.
        """
        n, nc2 = self.shape, comb(self.shape, 2)
        if x0 is None:
            x0 = np.random.uniform(low=0, high=2 * n, size=(n ** 2,))
        M = np.zeros((n, n))
        f_values = self.matrix.flatten()
        bnds = (n ** 2 - n) * [(bounds[0], bounds[1])] + n * [(0, np.inf)]
        solution = minimize(f_to_m, x0=x0, args=(f_values, n, conservative), method="SLSQP",
                            bounds=bnds)
        x = solution.x
        for i in range(n):
            start_ind = i * (n - 1)
            M[i, 0:i] = x[start_ind:start_ind + i]
            M[i, i + 1:n] = x[start_ind + i:start_ind + n - 1]
        return M, solution
