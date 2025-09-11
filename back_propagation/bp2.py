import torch
import time
import numpy as np

x_data=[0.0,1.0,2.0,3.0,4.0]
y_data=[3.00,4.56,6.22,-2.17,-48.19]#y=(-2)*exp(x)+3*(x**2)+2*x+5

a=torch.Tensor([-5.0])
a.requires_grad=True

b=torch.Tensor([1.0])
b.requires_grad=True

c=torch.Tensor([1.0])
c.requires_grad=True

d=torch.Tensor([3.0])
d.requires_grad=True

def forward(x):
  return a*(np.exp(x))+b*(x**2)+c*x+d
  
def loss(x,y):
  y_pred=forward(x)
  return (y_pred-y)**2
  
num=input("Please input your data:")
num_data=eval(num)
print("predict(before training):",num_data,forward(num_data).item())


for epoch in range(600):
  for x,y in zip(x_data,y_data):
    l=loss(x,y)
    l.backward()
    print("\tgrad:",x,y,a.grad.item(),b.grad.item(),c.grad.item(),d.grad.item())
    
    a.data-=0.0008*a.grad.data
    b.data-=0.002*b.grad.data
    c.data-=0.005*c.grad.data
    d.data-=0.02*d.grad.data
    
    a.grad.data.zero_()
    b.grad.data.zero_()
    c.grad.data.zero_()
    d.grad.data.zero_()
  print("epoch:",epoch+1,l.item())
  time.sleep(0.05)
print("predict(after training):",num_data,forward(num_data).item())
    
    