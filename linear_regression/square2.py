import torch
import time

#y=3*(x^2)-5*x+7
x_data=torch.Tensor([[-2.0],[-1.0],[0.0],[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[29.0],[15.0],[7.0],[5.0],[9.0],[19.0]])

class SquareModel(torch.nn.Module):
  def __init__(self):
    super(SquareModel,self).__init__()
    self.linear=torch.nn.Linear(2,1)
    
  def forward(self,x):
    x_sq=x**2
    x_f=torch.cat((x,x_sq),dim=1)
    
    y_pred=self.linear(x_f)
    return y_pred

model=SquareModel()

criterion=torch.nn.MSELoss(reduction='sum')
optimzer=torch.optim.SGD(model.parameters(),lr=0.0008)

for epoch in range(2000):
  y_pred=model(x_data)
  loss=criterion(y_pred,y_data)
  
  print("Epoch:",epoch+1,"Loss:",loss.item())
  
  optimzer.zero_grad()
  loss.backward()
  optimzer.step()
  time.sleep(0.2)
  
print("a=",model.linear.weight[0,1].item())
print("b=",model.linear.weight[0,0].item())
print("c=",model.linear.bias.item())

x_test=torch.Tensor([[-4.0],[-3.0],[2.5],[4.0]])
y_test=model(x_test)

print("x_test=",x_test.data,"\ny_test=",y_test.data)