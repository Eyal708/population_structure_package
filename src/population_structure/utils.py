import numpy as np
from .migration.migration import Migration
from .coalescence.coalescence import Coalescence
from .fst.fst import Fst
from .helper_funcs.helper_funcs import split_migration, reassemble_matrix


def find_fst(m: np.ndarray) -> np.ndarray:
    """
    Receives a migration matrix with one connected component(a squared, positive matrix with zeroes on the diagonal),
    and returns its corresponding Fst matrix according to Wilkinson-Herbot's equations and Slatkin's equations.
    :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
    :return: Corresponding Fst matrix according to Wilkinson-Herbot's equations. If there is no solution, an error will
    occur.
    """
    if m.shape[0] == 1:
        return np.array([[0]])
    M = Migration(m)
    t = M.produce_coalescence()
    T = Coalescence(t)
    return T.produce_fst()


def find_coalescence(m: np.ndarray) -> np.ndarray:
    """
       Receives a migration matrix with one connected component
       (a squared, positive matrix with zeroes on the diagonal), and returns its corresponding Coalescent times
       (T) matrix according to Wilkinson-Herbot's equations.
       :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
       :return: Corresponding T matrix according to Wilkinson-Herbot's equations. If there is no solution,
                an error will occur.
       """
    if m.shape[0] == 1:
        return np.array([[1]])
    M = Migration(m)
    return M.produce_coalescence()


def m_to_f(m: np.ndarray) -> np.ndarray:
    """
    Receives a migration matrix(a squared, positive matrix with zeroes on the diagonal) with any number
    of connected components, and returns its corresponding Fst matrix according to Wilkinson-Herbot's equations and
    Slatkin's equations.
    :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
    :return: Corresponding Fst matrix according to Wilkinson-Herbot's equations. If there is no solution, an error will
             occur.
    """
    split = split_migration(m)
    sub_matrices, components = split[0], split[1]
    f_matrices = []
    for matrix in sub_matrices:
        f_matrices.append(find_fst(matrix))
    return reassemble_matrix(f_matrices, components, "fst")


def m_to_t(m: np.ndarray) -> np.ndarray:
    """
       Receives a migration matrix(a squared, positive matrix with zeroes on the diagonal) with any number
       of connected components, and returns its corresponding Coalescent times (T) matrix according to
       Wilkinson-Herbot's equations.
       :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
       :return: Corresponding T matrix according to Wilkinson-Herbot's equations. If there is no solution,
                an error will occur.
       """
    split = split_migration(m)
    sub_matrices, components = split[0], split[1]
    t_matrices = []
    for matrix in sub_matrices:
        t_matrices.append(find_coalescence(matrix))
    return reassemble_matrix(t_matrices, components, "coalescence")


def m_to_t_and_f(m: np.ndarray) -> tuple:
    """
       Receives a migration matrix (a squared, positive matrix with zeroes on the diagonal) with any number
       of connected components, and returns its corresponding Coalescent times (T) matrix according to
       Wilkinson-Herbot's equations, and it's corresponding Fst matrix(F) according to Slatkin equations.
       :param m: Migration matrix- squared, positive, with zeroes on the diagonal.
       :return: A tuple (T,F). Corresponding T matrix according to Wilkinson-Herbot's equations,
                Corresponding F matrix according to Slatkin equations.
                If there is no solution, an error will occur.
       """
    split = split_migration(m)
    sub_matrices, components = split[0], split[1]
    t_matrices = []
    f_matrices = []
    for matrix in sub_matrices:
        t_matrix = find_coalescence(matrix)
        t_matrices.append(t_matrix)
        T = Coalescence(t_matrix)
        f_matrices.append(T.produce_fst())
    return reassemble_matrix(t_matrices, components, "coalescence"), reassemble_matrix(f_matrices, components, "fst")


def f_to_t(f: np.ndarray, x0=None, constraint=False, bounds=(0, np.inf)) -> tuple:
    """
    Gets an Fst matrix and generates a possible corresponding coalescence times matrix and returns it.
    Solves Slatkin's equations using a numerical solver.
    :param f: Fst matrix - squared, symmetric with values in range (0,1) matrix with zeroes on the diagonal.
    :param x0: initial guess for the variables, default is a random vector with bounds (0,2*n), where n is the
               size of the matrix (number of populations).
    :param constraint: indicated whether the T matrix produced should be 'compliant'. default is False
    :param bounds: bounds for each variable T(i,j), default is (0, inf). bounds should be a tuple of two arrays,
                  first is lower bounds for each variable, second is upper bounds for each variable.
                  If bounds is a tuple of two scalars, the same bounds are applied for each variable.
    :return: A tuple: (A possible corresponding Coalescence time matrix- T, details about
                      the solution of the numerical solver).
    """
    Fst_matrix = Fst(f)
    return Fst_matrix.produce_coalescence(x0, constraint, bounds)


def f_to_m(f: np.ndarray, x0=None, constraint=False, bounds_t=(0, np.inf), bounds_m=(0, 2), indirect=True) -> tuple:
    """
    Receives an Fst matrix and returns a possible corresponding migration matrix.
    Two approaches are available: the indirect approach, which first finds the coalescence matrix and then
    finds the corresponding migration matrix, and the direct approach, which finds the migration matrix
    directly from the Fst matrix.
    :param f: Fst matrix - squared, symmetric with values in range (0,1) matrix with zeroes on the diagonal.
    :param x0: initial guess for the variables, default is a random vector with bounds (0,2*n), where n is the
               size of the matrix (number of populations).
    :param constraint: indicated whether the T matrix produced should be 'compliant'. default is False. This is only
                       relevant when using the indirect approach (indirect=True).
    :param bounds_t: bounds for each variable T(i,j), default is (0, inf). bounds should be a tuple of two arrays,
                     first is lower bounds for each variable, second is upper bounds for each variable.
                     If bounds is a tuple of  two scalars, the same bounds are applied for each variable.
                     This is relevant when using the indirect approach.
    :param bounds_m: bounds for each individual variable. default is (0,2). bounds should be given as a tuple
                    of 2 arrays of size n**2-n (where n is the number of populations).
                    first array represents lower bounds, second array represents upper bounds. if a tuple with 2 scalars
                    is given instead, they will be the bounds for each variable.
    :param indirect: indicates whether to use the indirect approach of finding the coalescence matrix first, or the
                     direct approach of finding the migration matrix directly from the Fst matrix. default is True.
    :return: A tuple: first element is a possible corresponding migration matrix, second element is details about
                      the solution. If indirect is True, these are the details of the solution of the T->M
                      transformation which uses Linear Least Squares. Otherwise, these are the details of the numeric
                      solver that solves the F->M directly.
    """
    if indirect:
        T_matrix = Coalescence(f_to_t(f, x0, constraint, bounds_t)[0])
        return T_matrix.produce_migration(bounds_m)
    return Fst(f).produce_migration(x0, bounds_m)
