import time
import numpy as np

x_data=[-2.0,-1.0,0.0,1.0,2.0]
y_data=[1.27,1.73,3.0,6.44,15.78]

a=1.0
b=1.0

def forward(x):
  return a*(np.exp(x))+b
  
def loss(x,y):
  y_pred=forward(x)
  return (y_pred-y)**2
  
def agradient(x,y):
  return 2*(np.exp(x))*(a*(np.exp(x))+b-y)
  
def bgradient(x,y):
  return 2*(a*(np.exp(x))+b-y)

  
num=input("Please input your data:")
num_data=eval(num)  
print("Predict(before training):",num_data,forward(num_data))

for epoch in range(800):
  l=0
  
  for x_val,y_val in zip(x_data,y_data):
    y_pred_val=forward(x_val)
    a-=0.001*agradient(x_val,y_val)
    b-=0.002*bgradient(x_val,y_val)
    l+=loss(x_val,y_val)
    print("\tx={},y={},y_pred={},agrad={},bgrad={}".format(x_val,y_val,y_pred_val,agradient(x_val,y_val),bgradient(x_val,y_val)))
  print("Epoch:{},a={},b={},MSE={}".format(epoch+1,a,b,l/5))
  
  time.sleep(0.2)
print("Predict(after training):",num_data,forward(num_data))
