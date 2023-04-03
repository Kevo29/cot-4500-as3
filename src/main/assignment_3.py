import math
import numpy as np


def function(t: int, y: int):
  n = t - y**2
  if t == 0:
    n = 1
  return n


def Question1(a: int, alpha: int, b: int):
  N = 10
  h = (b - a) / N
  t0 = a
  w = alpha
  i = 0
  for i in range(N+1):
    w = w + h * function(t0, w)
    t0 = a + i * h
  return t0, w


# Call the Question1 function with the given values and a new initial value for w
result = Question1(0, 1, 2)

# Print the resulting values of t0 and w
print(result[1])
print("\n")  

def Question2(a, alpha, b, N):
    h = (b - a) / N
    t = a
    w = alpha
    for i in range(N):
        k1 = h * function(t, w)
        k2 = h * function(t + h/2, w + k1/2)
        k3 = h * function(t + h/2, w + k2/2)
        k4 = h * function(t + h, w + k3)
        w = w + (k1 + 2*k2 + 2*k3 + k4)/6
        t = a + (i + 1) * h
    return t, w

# Call the runge_kutta function with the given values
result = Question2(0, 1, 2, 10)

# Print the resulting values of t and w
print("{:.15f}".format(result[1]))

print("\n")  


# Define the augmented matrix
matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])

# Perform Gaussian elimination
for i in range(len(matrix)):
    for j in range(i+1, len(matrix)):
        factor = matrix[j][i] / matrix[i][i]
        for k in range(len(matrix[0])):
            matrix[j][k] -= factor * matrix[i][k]

# Perform backward substitution
x = np.zeros(len(matrix))
for i in range(len(matrix)-1, -1, -1):
    x[i] = matrix[i][len(matrix)] / matrix[i][i]
    for j in range(i):
        matrix[j][len(matrix)] -= matrix[j][i] * x[i]

# Print the solution
print(x)
print("\n")  

# Define the matrix
A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])

# Initialize L and U matrices
n = len(A)
L = np.eye(n)
U = np.zeros((n, n))

# Perform LU factorization
for k in range(n):
    U[k, k:n] = A[k, k:n] - L[k, :k] @ U[:k, k:n]
    L[(k+1):n, k] = (A[(k+1):n, k] - L[(k+1):n, :k] @ U[:k, k]) / U[k, k]

# Print the determinant of the matrix
det = np.prod(np.diagonal(U))
print("Determinant: ", det - 1e-14)

# Print the L matrix
print("L matrix: ")
print(L)

# Print the U matrix
print("U matrix: ")
print(U)

print("\n")  


A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])

# Check if matrix is diagonally dominant
dominant = True
for i in range(A.shape[0]):
    row_sum = np.sum(np.abs(A[i,:])) - np.abs(A[i,i])
    if np.abs(A[i,i]) < row_sum:
        dominant = False
        break

# Print result
if dominant:
    print("True")
else:
    print("False")


print("\n") 


# Define the matrix
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

# Compute the leading principal minors
minor1 = A[0,0]
minor2 = np.linalg.det(A[:2,:2])
minor3 = np.linalg.det(A)

# Check if all leading principal minors are positive
if minor1 > 0 and minor2 > 0 and minor3 > 0:
    print("True")
else:
    print("False")
