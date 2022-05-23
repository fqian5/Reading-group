import numpy as np
import scipy 
import pandas as pd
from scipy.linalg import hilbert
import sys
#np.set_printoptions(threshold=sys.maxsize)
def wilkinson(a, b, c):
	delta = (a-c)/2 #Define the value of delta
	return (c- (np.sign(delta)+ (delta == 0))* b**2 /(np.abs(delta) + (delta**2 + b**2)**.5))


def tridiag(A,):
	m ,n = A.shape
	W = np.zeros((m, n))
	for i in range(m-2):
		x = A[i+1:m,i]
		v = x + np.sign(x[0])*np.linalg.norm(x)*np.eye(len(x))[:m-i, 0]
		v = v/np.linalg.norm(v).reshape(-1,1)
		A[i+1:m, i:m] = A[i+1:m, i:m] - 2* np.matmul( v.T, np.matmul( v, A[i+1:m, i:m]))
		A[0:m, i+1:m] = A[0:m, i+1:m] - 2* np.matmul( np.matmul( A[0:m, i+1:m], v.T), v)
		W[i+1:m, i] = v.T.reshape(-1)
	R = A
	return (W, R)

H = np.eye(1024)
one = np.ones(1024)
H = H - 1/1024 * np.einsum('i, j -> ij', one, one)
A0   = pd.read_excel('A0.xlsx').to_numpy()
A    = np.dot(A0, H)
A    = np.dot(A, A.T)
T    = tridiag(A)[1]
eigh = []
j    = 0
t    = []

def house(A):
	m, n = A.shape #obtain the size of input matrix
	W = np.zeros((m,n))
	if m < n:
		raise Exception('Please input a tall matrix rather than a fat matrix')
	A0 = A #Store the original matrix
	for i in range(n):
		x = A[i:m,i]
		v = x + np.sign(x[0])*np.linalg.norm(x)*np.eye(len(x))[0]
		v = v/np.linalg.norm(v).reshape(-1,1)
		A[i:m, i:n] = A[i:m, i:n] - 2*  np.matmul( v.T, np.matmul( v, A[i:m, i:n]))
		W[i:m, i] = v.T.reshape(-1)
	R = A
	return W, R

def formQ(W, X):#In this situation plug in X = I to explicitly solve Q
	m, n = W.shape
	Q = X
	for j in range(n, -1, 1):
		w = W[j+1:n, j].reshape(-1)
		Q[j:n, :] = Q[j:n, :] -  2* np.matmul(w, np.matmul(w.T, Q[j:n, :]))
	return Q


def qralg(T):
	A = T
	tol = 1
	m, n = T.shape
	I = np.eye(T.shape[0])
	i = 0
	t = []
	while tol > 1e-12:
		T_old= T
		mu   = T[-1, -1]                                     #For simple shift
#		mu   = wilkinson(T[-2, -2], T[-1, -1], T[-2, -1])    #For wilkinson shift
#		Q, R = np.linalg.qr(T - mu * I)
		W, R = house(T - mu* I)
		Q    = formQ(W, I)
		T    = np.matmul(R, Q) + mu * I 
#		print(T)
#		print(np.linalg.norm(T-T_old))
		tol  = np.abs(T[-1, -2])
		i += 1
		t.append(np.abs(T[-1, -2]))
#		print(i, 'iterations')
	return(T, t)


while T.shape[0] > 1:
	j += 1
	T, t_temp = qralg(T)
	t.extend(t_temp)
	eigh.append(T[-1,-1])
	T = T[:-1, :-1]
eigh.append(T[-1,-1])


#print(np.linalg.eig(A))
print(eigh)
