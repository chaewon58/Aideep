import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x * x
y.backward()

print(y)        
print(x.grad)
