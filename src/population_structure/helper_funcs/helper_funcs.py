import math
import numpy as np
from collections import deque


def comb(n: int, k: int) -> int:
    """
    calculate and return n Choose k
    :param n: number of objects
    :param k: number of selected objects
    :return: n Choose k
    """
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def compute_coalescence(t: np.ndarray, f: np.ndarray, n: int) -> float:
    """
    returns the equations that describe the connection between coalescent times and Fst
    of all populations 1,2...n (Slatkin). These are the equations to minimize in order to find possible T matrices.
    :param t: an array representing [T(1,2),T(1,3)...,T(1,n),T(2,3)...,T(2,n),...T(1,1),T(2,2),...,T(n,n)], which are
              the variables to solve. Size of the array(number of unknowns) in nC2 + n.
    :param f: array of Fst values [F(1,2),F(1,3),,,,F(1,n),F(2,3),...F(2,n),...F(n-1,n)]. Array size is nC2.
    :param n: number of populations.
    :return: A list of all the equations that describe the connection between coalescent times and Fst
             of all populations 1,2...n (Slatkin).
    """
    eqs_lst = []
    nC2 = comb(n, 2)
    k = 0
    for i in range(nC2):
        for j in range(i + 1, n):
            eq = t[k] - (0.5 * (t[nC2 + i] + t[nC2 + j]) * ((1 + f[k]) / (1 - f[k])))
            eqs_lst.append(eq)
            k += 1
    return np.linalg.norm(eqs_lst)


def f_to_m(u: np.ndarray, f: np.ndarray, n: int) -> float:
    """
    Function to minimize in order to solve F->M directly (Xiran's paper).
    :param f: vector of Fst values (parameters) of size nC2.
    :param u: vector of unknown T and M values of size n^2.
    :param n: number of populations.
    :return: value of function at point u.
    """
    equation_lst = []
    m = u[:n ** 2 - n]  # M values
    t = u[n ** 2 - n:]  # T values
    for i in range(n):
        incoming_migrants_i = m[(n - 1) * i: (n - 1) * i + n - 1]  # only unknowns of kind M_{i,k}
        m_i = np.sum(incoming_migrants_i)
        other_t = np.concatenate((t[:i], t[i + 1:]))  # only unknowns of kind T_{k,k} where k!=i
        f_i_values = np.concatenate((f[n * i: n * i + i],
                                     f[n * i + 1 + i: n * i + n]))  # only f value of kind F_{i,k} where k!=i
        ones = np.ones(n - 1)
        f_i_vector = (ones + f_i_values) / (ones - f_i_values)
        equation_lst.append(((1 + m_i) * t[i] - 0.5 * (incoming_migrants_i @ ((other_t + t[i]) * f_i_vector))) - 1)
        for j in range(i):
            incoming_migrants_j = m[(n - 1) * j: (n - 1) * j + n - 1]
            m_j = np.sum(incoming_migrants_j)
            f_j_values = np.concatenate((f[n * j: n * j + i],
                                         f[n * j + 1 + i: n * j + n]))
            f_i_new_values = np.concatenate((f[n * i: n * i + j],
                                             f[n * i + 1 + j: n * i + n]))
            f_j_vector = (ones + f_j_values) / (ones - f_j_values)
            f_i_new_vector = (ones + f_i_new_values) / (ones - f_i_new_values)
            t_vals_no_j = np.concatenate((t[:j], t[j + 1:]))
            equation_lst.append(0.25 * (((m_i + m_j) * (t[i] + t[j]) * ((1 + f[(n * i + j)]) / (1 - f[(n * i + j)]))) -
                                        (incoming_migrants_i @ ((other_t + t[j]) * f_j_vector))
                                        - (incoming_migrants_j @ ((t_vals_no_j + t[i]) * f_i_new_vector))) - 1)
    return np.linalg.norm(equation_lst)


def constraint_generator(i: int, j: int) -> callable:
    """
    creates and returns a constraint function for the minimize algorithm used in F->T.
    :return: A callable which is a constraint function
    """

    def constraint(x: np.ndarray):
        return x[i] - x[j]

    return constraint


def cons_migration_constraint_generator(n: int, i: int) -> callable:
    """
    creates and returns a constraint function for the minimize algorithm used to directly solve F->M. this constraints
    are to assure conservative migration.
    :param n: total number of populations
    :param i: population number
    :return: A callable which is the conservative migration constraint for population i
    """

    def constraint(x: np.ndarray):
        m_values = x[:n ** 2 - n]
        indices = np.vstack(np.indices((n, n))).reshape(2, -1).T
        mask = indices[:, 0] != indices[:, 1]  # Off-diagonal mask
        m = np.zeros((n, n))
        m[tuple(indices[mask].T)] = m_values
        return np.sum(m[i, :]).round(2) - np.sum(m[:, i]).round(2)

    return constraint


def check_constraint(t: np.ndarray) -> bool:
    """
    gets a T matrix and returns True if it follows the within < inbetween constraint.
    :param t: Coalescence times matrix.
    :return: True if t follows the constraint, False otherwise.
    """
    min_indices = t.argmin(axis=1)
    diag_indices = np.diag_indices(t.shape[0])[0]
    return not np.any(min_indices != diag_indices)


def check_conservative(m: np.ndarray):
    """
    checks if a given migration matrix is conservative.
    :param m: A migration matrix
    :return: True if m is conservative, False otherwise.
    """
    for i in range(m.shape[0]):
        if np.sum(m[i, :]).round(2) != np.sum(m[:, i]).round(2):
            return False
    return True


def matrix_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the distance between two matrices. Matrices must be of the same shape.
    :param a: first matrix.
    :param b: second matrix.
    :return: The distance between a and b.
    """
    n = a.shape[0]
    c = np.abs(a - b)
    return float(np.sum(c)) / (n ** 2 - n)


def diameter(mats: list) -> float:
    """
    Calculates the diameter for a given set of matrices.
    :param mats: list containing a set of matrices of the same shape.
    :return: The diameter (maximum pair-wise distance) of the set of matrices 'mats'.
    """
    max_diam = 0
    for i in range(len(mats)):
        for j in range(i):
            max_diam = max(max_diam, matrix_distance(mats[i], mats[j]))
    return max_diam


def matrix_mean(mats: list) -> np.ndarray:
    """
    returns matrix which is the mean of a set of matrices, meaning each entry of the matrix is the mean of the entry
    across all matrices.
    :param mats: a set of matrices.
    :return: matrix mean.
    """
    return np.sum(mats, axis=0) / len(mats)


def find_components(matrix: np.ndarray) -> dict:
    """
    Find connected components in a directed graph represented by adjacency matrix.
    :param matrix: adjacency matrix representing a directed graph
    :return:something
    """
    components = 1
    n = matrix.shape[0]
    queue = deque()
    visited = set()
    not_visited = set([i for i in range(1, n)])
    visited.add(0)
    comp_dict = {components: [0]}
    queue.append(0)
    while len(not_visited) != 0:
        while len(queue) != 0:
            cur_vertex = queue.popleft()
            for i in range(n):
                if i not in visited and (matrix[cur_vertex, i] != 0 or matrix[i, cur_vertex] != 0):
                    queue.append(i)
                    visited.add(i)
                    not_visited.remove(i)
                    comp_dict[components].append(i)
        for vertex in not_visited:
            components += 1
            queue.append(vertex)
            visited.add(vertex)
            not_visited.remove(vertex)
            comp_dict[components] = [vertex]
            break
    return comp_dict


def split_migration_matrix(migration_matrix: np.ndarray, connected_components: list) -> list:
    """
    Splits a migration matrix to sub-matrices according to it's connected components.
    :param migration_matrix: A valid migration matrix.
    :param connected_components: list of lists, where each list represents a connected component's vertices
                                (populations).
    :return: A list of sub-matrices, where each sun-matrix is the migration matrix of a connected component. Note that
    in order to interpret which populations are described in each sub matrix the connected components list is needed.
    """
    sub_matrices = []
    for component in connected_components:
        sub_matrix = migration_matrix[np.ix_(component, component)]
        sub_matrices.append(sub_matrix)

    return sub_matrices


def split_migration(migration_matrix: np.ndarray) -> tuple:
    """
    Finds a migration matrix connected components, and splits the matrix to it's connected components.
    :param migration_matrix: A valid migration matrix.
    :return: A tuple (sub_matrices, components). Sub matrices is a list of numpy arrays, where each array is a
            component's migration matrix. components is a list of lists, where each list represents a component vertices
            (populations). The order of the components corresponds to the order of the sub-matrices.
    """
    components = list(find_components(migration_matrix).values())
    sub_matrices = split_migration_matrix(migration_matrix, components)
    return sub_matrices, components


def reassemble_matrix(sub_matrices: list, connected_components: list, which: str) -> np.ndarray:
    """
    Reassembles an Fst/Coalescence matrix according to sub-matrices and the connected components.
    :param sub_matrices: The sub matrices from which to assemble the matrix. A list of 2-D numpy arrays.
    :param connected_components: A list of lists, where each list is the connected components. Inidcates how the matrix
                                 should be assembled.
    :param which: Either "fst" or "coalescence". Indicated whether the assembled matrix is an Fst matrix or a
                  coalescence matrix. This is important for initialization of the returned matrix.
    :return: The assembled Fst or Coalescence matrix.
    """
    num_nodes = sum(len(component) for component in connected_components)
    if which == "fst":
        adjacency_matrix = np.ones((num_nodes, num_nodes), dtype=float)
    else:
        adjacency_matrix = np.full((num_nodes, num_nodes), np.inf)

    for component, sub_matrix in zip(connected_components, sub_matrices):
        indices = np.array(component)
        adjacency_matrix[np.ix_(indices, indices)] = sub_matrix

    return adjacency_matrix
