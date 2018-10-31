import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N = 25
X = np.reshape(np.linspace(0,0.9,N),(N,1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)
X1 = np.linspace(0,0.9,N)
Y1 = np.cos(10*X1**2) + 0.1*np.sin(100*X1)

string1 = "("
string2 = ","
string3 = ")"
n = [string1+str(round(X1[i],1))+string2+str(round(Y1[2],1))+string3 for i in range(25)]
fig, ax = plt.subplots()
ax.set_title('MLE - trigonometric basis function')
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
def omega_cal(order):
  phi = np.empty([N, 2*order+1])
  for i in range(N):
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
  y = np.reshape(np.cos(10*X**2) + 0.1*np.sin(100*X),(N,1))
  omega = np.dot(np.dot(inv(np.dot(phi_T, phi)), phi_T) , y) # shape=[3,1]
  #omega = np.reshape(omega,(order)) # shape=(3)
  #print(omega)
  return omega
#---------------------------------------
def f(t,order):
  omega = omega_cal(order)
  T = np.empty([num, 2*order+1])
  for i in range(num):
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
## try to plot MLE seperately
'''
num=100
t=np.linspace(-0.3,1.3,num)
y=f(t,order)
plt.plot(t, y, 'k')
plt.legend(('SAMPLE DATA','MLE'),
           loc='upper center', shadow=True)
'''
#---------------------------------------
## try to plot MLE in one picture
num=200
t=np.linspace(-1,1.2,num)
plt.plot(t, f(t,1), 'C0')
plt.plot(t, f(t,11), 'C4')
plt.legend(('SAMPLE DATA','MLE order 1','MLE order 11'),
           loc='upper center', shadow=True)
#---------------------------------------
plt.show()
