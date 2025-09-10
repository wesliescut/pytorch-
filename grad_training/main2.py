import time 

x_data=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
y_data=[14.0,28.0,48.0,74.0,106.0,144.0,188.0,238.0,294.0]

a=1.0
b=1.0
c=1.0

def forward(x):
  return a*x*x+b*x+c
  
def loss(x,y):
  y_pred=forward(x)
  return (y_pred-y)**2
  
def agradient(x,y):
  return 2*x*x*(a*x*x+b*x+c-y)
  
def bgradient(x,y):
  return 2*x*(a*x*x+b*x+c-y)
  
def cgradient(x,y):
  return 2*(a*x*x+b*x+c-y)

data=input("Please input your data:")
data_num=eval(data)  
print("Predict(before traning):",data_num,forward(data_num))

for epoch in range(800):
  l=0
  for x,y in zip(x_data,y_data):
    agrad=agradient(x,y)
    bgrad=bgradient(x,y)
    cgrad=cgradient(x,y)
    
    a=a-0.0001*agrad
    b-=0.0002*bgrad
    c-=0.0004*cgrad
    l+=loss(x,y)
    print("\tx={},y={},agrad={},bgrad={},cgrad={}".format(x,y,agrad,bgrad,cgrad))
  
  print("Epoch:",epoch+1,"a=",a,"b=",b,"c=",c,"MSE=",l/9)
  time.sleep(0.2)
print("Predict(after traning):",data_num,forward(data_num))
