from PIL import Image
import numpy as np
from random import randint
import math

def MyPCA(X,val):
    M = np.mean(X.T, axis=1)
    C = X - M
    V = np.cov(C.T)
    values,vectors = np.linalg.eig(V)
    
    x = np.argsort(values)
    x = x[::-1]
    
    vectors = vectors[:,x]
    values = values[x]
    if val >= 0 and val <= X.shape[1]:
        vectors = vectors[:,range(val)].real
    P = vectors.T.dot(C.T)
    return P.T

t = Image.open('original.png','r')
t = t.convert('1')
riv = []
pix_val = list(t.getdata())
pix_val = np.array(pix_val).reshape(512,512)

for i in range(512):
  for j in range(512):
    if(pix_val[i,j]==255):
      riv.append([i,j])

r = Image.open('1.gif','r')
g = Image.open('2.gif','r')
b = Image.open('3.gif','r')
a = Image.open('4.gif','r')

pix_val_r = list(r.getdata())
pix_val_r =np.array(pix_val_r).reshape(512,512)
pix_val_g = list(g.getdata())
pix_val_g =np.array(pix_val_g).reshape(512,512)
pix_val_b = list(b.getdata())
pix_val_b =np.array(pix_val_b).reshape(512,512)
pix_val_a = list(a.getdata())
pix_val_a =np.array(pix_val_a).reshape(512,512)

val = 4

X = np.array((pix_val_r,pix_val_g,pix_val_b,pix_val_a)).reshape(4,512*512).T
X = MyPCA(X,val)

x_nonriver = []
y_nonriver = []
for i in range(5000):
  x_nonriver.append(randint(1, 120))
  x_nonriver.append(randint(240, 500))

for i in range(10000):
  y_nonriver.append(randint(1,512))

x_river = []
y_river = []
for x in range(500):
  a = randint(0,len(riv))
  x_river.append(riv[a][0])
  y_river.append(riv[a][1])

pix_val_a[0]
def mean(X):
  return np.mean(X.T, axis=1)

river = []
for i in range(len(x_river)) :
  x = x_river[i]
  y = y_river[i]
  river = np.append(river,X[512*x+y]).reshape(-1,val)
  
nonriver = []

for i in range(len(x_nonriver)) :
  x = x_nonriver[i]
  y = y_nonriver[i]
  nonriver = np.append(nonriver,X[512*x+y]).reshape(-1,val)
  
n = river.shape[1]

river_mean = mean(river)
nonriver_mean = mean(nonriver)

cov_mat_river = np.cov(river.T).reshape(-1,val)
cov_mat_nonriver= np.cov(nonriver.T).reshape(-1,val)

test_data= X.T
test_data.shape

out= []
for i in range(0,512*512):
  river_class = np.matmul(np.matmul((((test_data[:,i]-river_mean).reshape(n,1)).T),(np.linalg.inv(cov_mat_river))),(((test_data[:,i]-river_mean).reshape(n,1))))
  nonriver_class = np.matmul(np.matmul((((test_data[:,i]-nonriver_mean).reshape(n,1)).T),(np.linalg.inv(cov_mat_nonriver))),(((test_data[:,i]-nonriver_mean).reshape(n,1))))
  if (river_class > 709):
    river_class = 0
  p1 = (-0.5)*1/math.sqrt(np.linalg.det(cov_mat_river))*(math.exp(river_class))
  P2 = 0.09
  p2 = (-0.5)*1/math.sqrt(np.linalg.det(cov_mat_nonriver))*math.exp(nonriver_class)
  P1 = 1-P2
  if (p1*P1 > p2*P2):
    out.append(255)
  else :
    out.append(0)


out = np.array(out,dtype=np.uint8).reshape(512,512)
new_image = Image.fromarray(out)

new_image.show()
