import torch
import time

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[2.0], [9.0], [20.0], [35.0]])



class SimpleQuadraticModel(torch.nn.Module):
    def __init__(self):
        super(SimpleQuadraticModel, self).__init__()
      
        self.linear = torch.nn.Linear(2, 1)  
        
    def forward(self, x):
        
        x_squared = x ** 2
       
        x_features = torch.cat((x, x_squared), dim=1)
        y_pred = self.linear(x_features)
        return y_pred


model = SimpleQuadraticModel()  

criterion = torch.nn.MSELoss(reduction='sum') 
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    print("EPOCH:", epoch+1, "loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    time.sleep(0.1)
    

if isinstance(model, SimpleQuadraticModel):
    print("w1:", model.linear.weight[0, 0].item())
    print("w2:", model.linear.weight[0, 1].item())
    print("b:", model.linear.bias.item())
else:
    
    print("NO")

x_test = torch.Tensor([[5.0], [6.0]])
y_test = model(x_test)

print("x_test=", x_test.data, "\ny_pred=", y_test.data)