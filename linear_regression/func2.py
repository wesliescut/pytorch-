import torch
import time

#y=2*x^2-5*x-2*ln(x)+3
x_data=torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]])
y_data=torch.Tensor([[0.00],[-0.39],[3.80],[12.23],[24.78],[41.42]])

class FuncModel(torch.nn.Module):
  def __init__(self):
    super(FuncModel,self).__init__()
    self.linear=torch.nn.Linear(3,1)
    
  def forward(self,x):
    x_s=x**2
    x_l=torch.log(x)
    
    x_f=torch.cat((x_s,x,x_l),dim=1)
    y_pred=self.linear(x_f)
    return y_pred
    
model=FuncModel()

criterion=torch.nn.MSELoss(reduction='sum')
optimzer=torch.optim.SGD(model.parameters(),lr=0.0003)

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
print("d=",model.linear.bias.item())

x_test=torch.Tensor([[1.5],[2.4],[3.2]])
y_test=model(x_test)

print("x_test=",x_test.data,"\ny_test=",y_test.data)
  