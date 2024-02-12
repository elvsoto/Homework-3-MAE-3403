import math
import copy

# GaussSeidel from HW2
def GaussSeidel(Aaug):
    """
    Use the Gauss-Seidel method to estimate the solution to a set of linear equations Ax = b.
    :param Aaug: An augmented matrix containing [A|b] having N rows and N+1 columns, where N is the number of equations in the set.
    :return: the final new x vector.
    """

    N = len(Aaug)

    # check and modify matrix to diagonally dominant

    for k in range(15):
        x_prev = copy.deepcopy(Aaug)
        for i in range(N):
            sigma = sum(Aaug[i][j] * x_prev[j] for j in range(N) if j != i)
            Aaug[i][-1] = (Aaug[i][-1] - sigma) / Aaug[i][i]

    return Aaug

# Check symmetry
def is_symmetric(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    if rows != columns:  # check if the matrix is square
        return False
    # check symmetry by comparing every element with its transpose
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


# Check positive definite (use determinant.py from Dr.Smay)
def GetMinorMatrix(matrix, row, column):
    '''
    This removes a row and column from a deep copy of matrix
    :param A: original matrix
    :param row: row index to remove
    :param column: column index to remove
    :return: modified matrix
    '''

    newMatrix = copy.deepcopy(matrix)  # make a deep copy of matrix
    newMatrix.pop(row)  # removes row from matrix
    for r in newMatrix:  # removes column from matrix
        r.pop(column)
    return newMatrix


def Determinant(matrix):
    '''
    Compute the determinant of a nXn matrix using 1st row for cofactors.
    Note: Detrminant calls itself if the matrix is larger than 1x1.
    :param matrix: matrix to take a derivative of
    :return:determinant of matrix (scalar)
    '''

    det = 0
    if len(matrix) == 1:  # smallest possible minor matrix
        return matrix[0][0]
    else:
        for c in range(len(matrix[0])):  # using the first row for cofactors
            sign = (-1) ** c  # calculate the sign as presicribed by determinant using cofactors and minors method
            if (matrix[0][c] != 0):  # if cofactor is zero, no contribution to determinant
                det += sign * matrix[0][c] * Determinant(GetMinorMatrix(matrix, 0, c))  # recursively solve determinant
    return det


def is_positive_definite(matrix):
    # Check symmetry
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    # Check positive definiteness using principal minors
    for k in range(1, n + 1):
        minor = [row[:k] for row in matrix[:k]]
        if Determinant(minor) <= 0:
            return False
    return True

# Doolittle's LU Factorization Functions

def LUFactorization(A):
    """
    This is the Lower-Upper factorization part of Doolittle's method.  The factorization follows the work in
    Kreyszig section 20.2.  Note: L is the lower triangular matrix with 1's on the diagonal.  U is the upper traingular matrix.
    :param A: a nxn matrix
    :return: a tuple with (L, U)
    """
    n = len(A)
    # Step 1
    U = [([0 for c in range(n)] if not r == 0 else [a for a in A[0]]) for r in range(n)]
    L = [[(1 if c == r else (A[r][0] / U[0][0] if c == 0 else 0)) for c in range(n)] for r in range(n)]

    # step 2
    for j in range(1, n):  # j is row index
        # (a)
        for k in range(j, n):  # always j >= 1 (i.e., second row and higher)
            U[j][k] = A[j][k]  # k is column index and scans from column j to n-1
            for s in range(j):  # s is column index for L and row index for U
                U[j][k] -= L[j][s] * U[s][k]
            # (b)
            for i in range(k + 1, n):
                sig = 0
                for s in range(k):
                    sig += L[i][s] * U[s][k]
                L[i][k] = (1 / (U[k][k])) * (A[i][k] - sig)
    return (L, U)


def BackSolve(A, b, UT=True):
    """
    This is a backsolving algorithm for a matrix and b vector where A is triangular
    :param A: A triangularized matrix (Upper or Lower)
    :param b: the right hand side of a matrix equation Ax=b
    :param UT: boolean of upper triangular (True) or lower triangular (False)
    :return: the solution vector x, from Ax=b
    """
    nRows = len(b)
    x = [0] * nRows
    if UT:
        for nR in range(nRows - 1, -1, -1):
            s = 0
            for nC in range(nR + 1, nRows):
                s += A[nR][nC] * x[nC]
            x[nR] = 1 / A[nR][nR] * (b[nR] - s)
    else:
        for nR in range(nRows):
            s = 0
            for nC in range(nR):
                s += A[nR][nC] * x[nC]
            x[nR] = 1 / A[nR][nR] * (b[nR] - s)
    return x


def Doolittle(Aaug):
    """
    :param Aaug: the augmented matrix
    :return: the solution vector x
    """
    A, b = GaussSeidel(Aaug)
    L, U = LUFactorization(A)
    y = BackSolve(L, b, UT=False)
    x = BackSolve(U, y, UT=True)
    return x


# Cholesky's Method
def Cholesky(A):
    """
    Cholesky decomposition of a symmetric positive definite matrix A
    :param A: Symmetric positive definite matrix
    :return: Lower triangular matrix L such that A = LL^T
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i+1):
            if i == j:
                sum_val = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = math.sqrt(A[i][i] - sum_val)
            else:
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum_val))
    return L

def CholeskySolver(A, b):
    """
    Solves the system Ax = b using Cholesky decomposition
    :param A: Symmetric positive definite matrix A
    :param b: Right-hand side vector b
    :return: Solution vector x
    """
    L = Cholesky(A)
    LT = transpose(L)
    y = BackSolve(L, b, UT=False)
    x = BackSolve(LT, y, UT=True)
    return x

def transpose(matrix):
    """
    Transposes a given matrix
    :param matrix: Matrix to be transposed
    :return: Transposed matrix
    """
    return [list(row) for row in zip(*matrix)]

def main():
    A1 = [[1,-1,3,2],
         [-1,5,-5,-2],
         [3,-5,19,3],
         [2,-2,3,21]]
    b1 = [15,-35,94,1]

    if is_symmetric(A1) and is_positive_definite(A1):
        x1 = CholeskySolver(A1, b1)
        print("Solution for Matrix 1: ", x1)
        print("Method used: Cholesky")
    else:
        x1 = Doolittle(A1)
        print("Solution for Matrix 1: ", x1)
        print("Method used: Doolittle")

    A2 = [[4, 2, 4, 0],
          [2, 2, 3, 2],
          [4, 3, 6, 3],
          [0, 2, 3, 9]]
    b2 = [20, 36, 60, 12]

    if is_symmetric(A2) and is_positive_definite(A2):
        x2 = CholeskySolver(A2, b2)
        print("Solution for Matrix 2: ", x2)
        print("Method used: Cholesky")
    else:
        x2 = Doolittle(A2)
        print("Solution for Matrix 2: ", x2)
        print("Method used: Doolittle")

if __name__ == '__main__':
    main()

