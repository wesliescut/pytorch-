import torch
import time
import numpy as np

x_data=[1.0,2.0,3.0,4.0,5.0,6.0]
y_data=[3.56,16.53,43.73,69.42,30.56,-264.65]#y=2*(x**3)+3*(x**2)+4-2*(np.exp(x))-(np.log(x))

a=torch.Tensor([1.0])
a.requires_grad=True

b=torch.Tensor([2.0])
b.requires_grad=True

c=torch.Tensor([2.0])
c.requires_grad=True

d=torch.Tensor([-1.0])
d.requires_grad=True

e=torch.Tensor([-1.0])
e.requires_grad=True

def forward(x):
  return a*(x**3)+b*(x**2)+c+d*(np.exp(x))+e*(np.log(x))
  
def loss(x,y):
  y_pred=forward(x)
  return (y_pred-y)**2
  
num=input("Please input your data:")
num_data=eval(num)
print("Predict(before training):",num_data,forward(num_data).item())

for epoch in range(2000):
  for x,y in zip(x_data,y_data):
    l=loss(x,y)
    l.backward()
    
    print("\tgrad:",x,y,a.grad.item(),b.grad.item(),c.grad.item(),d.grad.item(),e.grad.item())
    
    a.data-=0.000004*a.grad.data
    b.data-=0.000008*b.grad.data
    c.data-=0.005*c.grad.data
    d.data-=0.000008*d.grad.data
    e.data-=0.005*e.grad.data
    
    a.grad.data.zero_()
    b.grad.data.zero_()
    c.grad.data.zero_()
    d.grad.data.zero_()
    e.grad.data.zero_()
    
  print("Epoch:",epoch+1,l.item())
  time.sleep(0.5)

print("Predict(after training):",num_data,forward(num_data).item())
    