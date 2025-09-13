import torch
import time

#y=e^x-2*x^3+3*x^2+4*x-1
x_data=torch.Tensor([[-3.0],[-2.0],[-1.0],[0.0],[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[68.04],[19.14],[0.37],[0.00],[6.72],[10.39],[4.09]])

class FuncModel(torch.nn.Module):
  def __init__(self):
    super(FuncModel,self).__init__()
    self.linear=torch.nn.Linear(4,1)
    
  def forward(self,x):
    x_exp=torch.exp(x)
    x_c=x**3
    x_s=x**2
    
    x_f=torch.cat((x_exp,x_c,x_s,x),dim=1)
    y_pred=self.linear(x_f)
    return y_pred
    
model=FuncModel()

criterion=torch.nn.MSELoss(reduction='sum')
optimzer=torch.optim.SGD(model.parameters(),lr=0.0001)


for epoch in range(20000):
  y_pred=model(x_data)
  loss=criterion(y_pred,y_data)
  
  print("Epoch:",epoch+1,"Loss:",loss.item())
  
  optimzer.zero_grad()
  loss.backward()
  optimzer.step()
  
  time.sleep(0.1)
  
print("a=",model.linear.weight[0,0].item())
print("b=",model.linear.weight[0,1].item())
print("c=",model.linear.weight[0,2].item())
print("d=",model.linear.weight[0,3].item())
print("e=",model.linear.bias.item())

x_test=torch.Tensor([[-4.0],[4.0]])
y_test=model(x_test)

print("x_test=",x_test.data,"\ny_test=",y_test.data)