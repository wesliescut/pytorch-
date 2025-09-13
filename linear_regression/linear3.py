import torch
import time

x_data=torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y_data=torch.Tensor([[1.0],[-1.0],[-3.0],[-5.0],[-7.0]])

class LinearModel(torch.nn.Module):
  def __init__(self):
    super(LinearModel,self).__init__()
    self.linear=torch.nn.Linear(1,1)
    
  def forward(self,x):
    y_pred=self.linear(x)
    return y_pred
    
model=LinearModel()
criterion=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
  y_pred=model(x_data)
  loss=criterion(y_pred,y_data)
  
  print("Epoch:",epoch+1,"Loss:",loss.item())
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  time.sleep(0.2)
  
print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())

x_test=torch.Tensor([[-3.0],[-1.0],[6.0]])
y_test=model(x_test)
print("x_test=",x_test.data,"\ny_test=",y_test.data)