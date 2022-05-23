import numpy as np
import scipy
import matplotlib.pyplot as plt

m = 80 #Define the dimension of matrix A
n = 80

if m < n:
	raise Exception('m should be larger than n')
'''
a = 0.7
b = 0.8
delta = 1e-11
#Define constants

A = np.random.rand(m,n) #Obtain random matrix A
A = np.asarray(((a, b),(a + delta,b)))  # matrix in 9.1
'''

def QR_fac(mat):
	m, n = mat.shape
	Q = np.zeros((m,m))
	R = np.zeros((m,n))
	V = np.zeros((m,n)) 
	for j in range(n):
		V[:,j] = mat[:,j]
		for i in range(j):
			R[i,j] = np.dot(Q[:,i].T, mat[:,j])
			V[:,j] = V[:,j] - R[i,j]* Q[:,i]
		R[j,j] = np.linalg.norm(V[:,j])
		Q[:,j] = V[:,j] / R[j,j]

	return (Q, R) 

A = np.random.rand(m,n) #Obtain random matrix A
Sigma_diag = np.zeros(80)
for i in range(80):
	Sigma_diag[i] = 2**(-i-1)
Sigma = np.diag(Sigma_diag)

Q, R = QR_fac(A) #obtain an orthogonal basis set
A = np.random.rand(m,n)
Q1, R = QR_fac(A) #obtain another orthogonal basis set
A = np.matmul(Q, Sigma, Q1)

Q, R = QR_fac(Sigma) #Find the QR factorization of A

print('L-2 norm\n',  np.linalg.norm(np.matmul(Q.T, Q)-np.identity(80))) #check the orthonormality
print('Frobenius norm\n',  np.linalg.norm(np.matmul(Q.T, Q)-np.identity(80), 'fro')) #check the orthonormality
print('A\n', A, '\n', 'QR\n', np.matmul(Q,R))
print('Q\n', Q, '\n', 'R\n', R)
	
