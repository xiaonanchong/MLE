import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#order =2
N = 25

X = np.reshape(np.linspace(0,0.9,N),(N,1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

fig, ax = plt.subplots()
ax.set_title('leave_one_out cross validation')
ax.set_xlabel('order')
ax.set_ylabel('average test square error')
#--------------------------------------
## use 35-1 sample data to extimate the function parameter

def omega_cal(order,X,M):
  phi = np.empty([M, 2*order+1])
  for i in range(M):
    for j in range(2*order+1):
      if(j==0):
        phi[i][j]=1
      elif(j%2 ==1):
        J = (j+1)/2
        phi[i][j]=np.sin(2*np.pi*J*X[i][0])
      else:
        J = j/2
        phi[i][j]=np.cos(2*np.pi*J*X[i][0])
  #print(phi)
  phi_T = np.transpose(phi)
  y = np.reshape(np.cos(10*X**2) + 0.1*np.sin(100*X),(M,1))
  omega = np.dot(np.dot(inv(np.dot(phi_T, phi)), phi_T) , y) # shape=[3,1]
  #omega = np.reshape(omega,(order)) # shape=(3)
  #print(omega)
  return omega

#---------------------------------------
def f(t,order,X,M):
  omega = omega_cal(order,X,M)
  T = np.empty([1, 2*order+1])
  for i in range(1):
    T[i]=append(t[i],order)
  return np.dot(T,omega)

def append(t,order):
  T = np.empty([1, 2*order+1])
  for j in range(2*order+1):
    if(j==0):
      T[0][j]=1
    elif(j%2 ==1):
      J = (j+1)/2
      T[0][j]=np.sin(2*np.pi*J*t)
    else:
      J = j/2
      T[0][j]=np.cos(2*np.pi*J*t)
  return T
#---------------------------------------
## validate on the last sample data
def cross_validation(order,M):
  sum_error = 0
  for i in range(N):
    X_M = np.delete(X, (i), axis=0)
    error = f(X[i], order,X_M,M)-(np.cos(10*X[i][0]**2) + 0.1*np.sin(100*X[i][0]))
    square_error = error**2
    sum_error = sum_error+square_error

  average_square_error = sum_error/N
  return average_square_error
#---------------------------------------
##sigma square by MLE
def sigma_square(order,M):
  
  sum_error = 0
  for i in range(N):
    error = f(X[i], order,X,M)-(np.cos(10*X[i][0]**2) + 0.1*np.sin(100*X[i][0]))
    square_error = error**2
    sum_error = sum_error+square_error

  average_square_error = sum_error/N
  return average_square_error
#---------------------------------------
print(cross_validation(2,N-1))
print(sigma_square(2,N))
#---------------------------------------
## try to plot error for difference orders in one picture
XX = [0,1, 2, 3, 4,5,6,7,8,9,10]
YY = np.empty([11])
sigma = np.empty([11])
for i in range(11):
  YY[i] = cross_validation(i,N-1)
for i in range(11):
  sigma[i] = sigma_square(i,N)
plt.plot(XX, YY)
plt.plot(XX, sigma)
plt.plot(XX,YY, 'r+')
plt.plot(XX, sigma, 'g^')
##set annotation
notation = [str(round(YY[i],2)) for i in range(11)]
for i, x in enumerate(notation):
    ax.annotate(notation[i], (XX[i], YY[i]), fontsize=12)
notation = [str(round(sigma[i],2)) for i in range(11)]
for i, x in enumerate(notation):
    ax.annotate(notation[i], (XX[i], sigma[i]), fontsize=12)
plt.legend(('cross validation','ML value for sigma square'),
           loc='upper center', shadow=True)
#---------------------------------------
plt.show()
