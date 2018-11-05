import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

L = 0.1
order = 20
mu = np.linspace(0,1, 20)
#print(mu)
lam = 0.5

N = 25
X = np.reshape(np.linspace(0,0.9,N),(N,1)) # [N,1]
#print(X)
Y = np.cos(10*X**2) + 0.1*np.sin(100*X) # [N,1]
X1 = np.linspace(0,0.9,N) # [N]
Y1 = np.cos(10*X1**2) + 0.1*np.sin(100*X1) # [N]

string1 = "("
string2 = ","
string3 = ")"
n = [string1+str(round(X1[i],1))+string2+str(round(Y1[i],1))+string3 for i in range(25)]
fig, ax = plt.subplots()
ax.set_title('Ridge regression - Gaussian basis function')
ax.set_xlabel('X axes')
ax.set_ylabel('Y axes')
ax.set_xlim([-0.3,1.3])
ax.set_ylim([-1.2,2.6])
#ax.scatter(X,Y)
plt.plot(X,Y, 'r+', label = "Training data")
## set annotation
for i, x in enumerate(n):
    ax.annotate(n[i], (X[i], Y[i]), fontsize=8)
## set legend label
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#red_patch = mpatches.Patch(color='red', label='Training data')
#plt.legend(handles=[red_patch])
#--------------------------------------
def omega_cal(order, lam):
  phi = np.empty([N, order+1])
  for i in range(N):
    for j in range(order+1):
      if(j==0):
        phi[i][j]=1
      else:
        #print(i,j)
        phi[i][j]=np.exp(-(X[i][0]-mu[j-1])**2 / (2*L**2) )
  
  phi_T = np.transpose(phi)
  #print(phi_T)
  y = np.reshape(np.cos(10*X**2) + 0.1*np.sin(100*X),(N,1))
  addition = lam * np.identity(order +1)
  omega = np.dot(np.dot(inv(np.dot(phi_T, phi) + addition), phi_T) , y)
  #print(omega)
  return omega
#---------------------------------------
def f(t,order, lam):
  omega = omega_cal(order, lam)
  T = np.empty([num, order+1])
  for i in range(num):
    T[i]=append(t[i],order)
  print(T)
  #print(np.dot(T,omega))
  return np.dot(T,omega)

def append(t,order):
  T = np.empty([1, order+1])
  for j in range(order+1):
    if(j==0):
      T[0][j]=1
    else:
      T[0][j]=np.exp(-(t-mu[j-1])**2 / (2*L**2))
  return T
#---------------------------------------
## try to plot MLE in one picture
num=200
t=np.linspace(-0.3,1.3,num)
plt.plot(t, f(t,order, 18), 'C0')
plt.plot(t, f(t,order, 0.001), 'C1')
plt.plot(t, f(t,order, 0), 'C2')
plt.legend(('SAMPLE DATA','unerfitting lambda=18','reasonable fit lambda=0.001','overfitting lambda=0'),
           loc='upper center', shadow=True)
#---------------------------------------
plt.show()
