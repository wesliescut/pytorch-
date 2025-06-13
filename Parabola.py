import numpy as np
import torch

x_data=[0.0,1.0,1.5,2.0,3.0]
y_data=[7.0,16.0,22.75,31.0,52.0]

a=torch.Tensor([1.0])
b=torch.Tensor([2.0])
c=torch.Tensor([3.0])
a.requires_grad=True
b.requires_grad=True
c.requires_grad=True

def forward(x):
  return a*(x**2)+b*x+c
  
def cost(x,y):
  y_pred=forward(x)
  return (y_pred-y)**2
  
print('Predict(before training)',4,forward(4).item())
for epoch in range(500):
  for x,y in zip(x_data,y_data):
    cost_val=cost(x,y)
    cost_val.backward()
    print("\tgrad:{},{},{},{},{}".format(x,y,a.grad.item(),b.grad.item(),c.grad.item()))
    a.data-=0.01*a.grad.data
    b.data-=0.01*b.grad.data
    c.data-=0.01*c.grad.data
    
    a.grad.zero_()
    
    b.grad.zero_()
    c.grad.zero_()
   
  
  
  print("Epoch:{},a={},b={},c={},loss={}".format(epoch+1,a.data.item(),b.data.item(),c.data.item(),cost_val.item()))
print('Predict(after training)',4,forward(4).item())