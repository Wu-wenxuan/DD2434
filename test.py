import numpy as np
#a=np.array([[0,2,3],[2,0,6],[3,6,0]])
a=np.mat([[2,-1,0],[-1,2,-1],[0,-1,2]])
b=np.mat([1,1,1])
R=a*b.T*b/a.shape[1]
C=b.T*b*a/a.shape[1]
R_C=b.T*b*a*b.T*b/np.power(a.shape[1],2)

a1=np.mat(np.ones((3,3)))
# b=np.ones(a.shape[0])
# R=a*b*b.T/a.shape[0]
# C=b*b.T*a/a.shape[0]
# R_C=b*b.T*a*b*b.T/np.power(a.shape[0],2)
# mean=np.mean(a)
S= -1/2*(a-R-C+R_C)

A=np.eye(4)
C1 = np.linalg.cholesky(S)
print(C1)

B = np.linalg.eigvals(S)
print(B)
# a2=np.array([1,2,3,4])
# print(np.sqrt(np.sum(np.power(a2,2))))
# D=np.array([[0]for i in range(a.shape[0])])
# D[0,0] = 1
# print(D)
# c=np.ones(3)
# print(np.mat(c).T)

