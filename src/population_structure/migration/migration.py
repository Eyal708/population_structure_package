import numpy as np
from ..helper_funcs.helper_funcs import comb as comb
from importlib_resources import files
import ctypes


class Migration:
    """
    A class that represents a migration matrix and provides methods to produce the corresponding
    coalescence matrix. The production of the coalescence matrix is according to Wilkinson-Herbot's 2003 paper.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initialize a migration matrix object
        :param matrix: input migration matrix
        """
        self.matrix = matrix.astype(float)
        self.shape = matrix.shape[0]
        self.lib = None

    def load_c_library(self) -> None:
        """
        Loads the C library that calculates the coefficient matrix.
        """
        lib = ctypes.cdll.LoadLibrary(str(files('population_structure.data').joinpath('libmigration_noGSL.dll')))
        lib.coefficient_matrix_from_migration.restype = ctypes.POINTER(ctypes.c_double)
        lib.coefficient_matrix_from_migration.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.lib = lib

    def produce_coalescence_old(self) -> np.ndarray:
        """
        produces and returns the corresponding coalescence matrix. This is the old method that doesn't use the
        coefficient_matrix_from_migration_wrapper method, which means it is less efficient.
        :return: The corresponding coalescence matrix
        """
        A = self.produce_coefficient_matrix()
        b = self.produce_solution_vector()
        x = np.linalg.solve(A, b)
        T_mat = np.zeros((self.shape, self.shape))
        # Assign values from x to T_mat using the upper triangular indices
        i, j = np.triu_indices(self.shape)
        T_mat[i, j] = x
        T_mat[j, i] = x
        return T_mat

    def produce_coalescence(self) -> np.ndarray:
        """
        produces and returns the corresponding coalescence matrix. This is the efficient method that uses the
        coefficient_matrix_from_migration_wrapper method (runs in C).
        :return: The corresponding coalescence matrix
        """
        A = self.coefficient_matrix_from_migration_wrapper()
        b = self.produce_solution_vector()
        x = np.linalg.solve(A, b)
        T_mat = np.zeros((self.shape, self.shape))
        i, j = np.triu_indices(self.shape)
        T_mat[i, j] = x
        T_mat[j, i] = x
        return T_mat

    def coefficient_matrix_from_migration_wrapper(self) -> np.ndarray:
        """
        Wrapper for the C function that calculates the coefficient matrix from the migration matrix.
        :return: the coefficient matrix corresponding to the migration matrix.
        """
        self.load_c_library()  # load the C library
        n = self.shape
        mat_size = n + (n * (n - 1)) // 2  # size of the coefficient matrix
        migration_matrix_c = self.matrix.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result_c = self.lib.coefficient_matrix_from_migration(migration_matrix_c, n)
        result = np.ctypeslib.as_array(result_c, shape=(mat_size * mat_size,)).reshape((mat_size, mat_size))
        return result

    def calculate_first_coefficients(self, j: int, i: int, same_pop: int, lower_bound: int, upper_bound: int,
                                     p_list: list, counter: list) -> float:
        """
        calculates the coefficients for the first n equations
        :param j: column of coefficient matrx
        :param i: row of coefficient matrix
        :param same_pop: The column corresponding to T(i,i)
        :param lower_bound: j values in range [lower_bound, upper_bound] correspond to coefficient -M(i,i+counter)
        :param upper_bound: j values in range [lower_bound, upper_bound] correspond to coefficient -M(i,i+counter)
        :param p_list: all values that are smaller than i
        :param counter: counts the number of time j was in the interval [lower_bound,upper_bound]
        :return: The coefficient in place [i,j], for i in [n-1]
        """
        n = self.matrix.shape[0]
        if j == same_pop:
            return 1 + np.sum(self.matrix[i, :])
        if lower_bound <= j <= upper_bound:
            counter[0] += 1
            return -1 * self.matrix[i, i + counter[0] - 1]
        for p in p_list:
            if j == (i - p) + np.sum([n - k for k in range(p)]):
                return -1 * self.matrix[i, p]
        return 0

    def calculate_last_coefficients(self, j, cur_pop, other_pop) -> float:
        """
        calculates the coefficients for the last (n choose 2) rows in the coefficient matrx
        :param j: the column in the coefficient matrix
        :param cur_pop: the index of the population that corresponds to the current value
        :param other_pop: the index of the other population that corresponds to the current value
        :return: The coefficient in the coefficient matrix according to certain conditions deduced from
        Wilkinson-Herbots' equations.
        """
        n = self.matrix.shape[0]
        if j == np.sum([n - k for k in range(other_pop)]) + (cur_pop - other_pop):
            return float(np.sum(self.matrix[[cur_pop, other_pop], :]))
        for p in range(n):
            for t in [other_pop, cur_pop]:
                if t == other_pop:
                    not_t = cur_pop
                else:
                    not_t = other_pop
                if p != not_t:
                    min_t_p = min(t, p)
                    max_t_p = max(t, p)
                    if j == np.sum([n - k for k in range(min_t_p)]) + max_t_p - min_t_p:
                        return -1 * self.matrix[not_t, p]
        return 0

    def produce_coefficient_matrix(self) -> np.ndarray:
        """
        produce and return the coefficient matrix used to calculate the T matrix(coalescence).
        :return: Coefficient matrix corresponding to object's migration matrix
        """
        n = self.shape
        n_last_equations = comb(n, 2)
        n_first_equations = n
        mat_size = n_first_equations + n_last_equations
        coefficient_mat = np.zeros((mat_size, mat_size))
        for i in range(n_first_equations):
            same_population = int(np.sum([n - k for k in range(i)]))
            lower_bound = same_population + 1
            upper_bound = np.sum([n - k for k in range(i + 1)]) - 1
            smaller_ind_lst = [p for p in range(i)]
            counter = [1]
            for j in range(mat_size):
                coefficient_mat[i, j] = self.calculate_first_coefficients(j, i, same_population, lower_bound,
                                                                          upper_bound, smaller_ind_lst, counter)
        cur_population = 1
        other_population = 0
        for i in range(n_last_equations):
            if other_population == cur_population:
                other_population = 0
                cur_population += 1
            for j in range(mat_size):
                coefficient_mat[n + i, j] = self.calculate_last_coefficients(j, cur_population, other_population)
            other_population += 1

        return coefficient_mat

    def produce_solution_vector(self):
        """
        produce the solution vector(b), according to Wilkinson-Herbot's equations
        :return: solution vector b
        """
        n = self.shape
        n_first = np.repeat(1, n)
        n_last = np.repeat(2, comb(n, 2))
        return np.hstack((n_first, n_last))
