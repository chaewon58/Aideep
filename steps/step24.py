if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

# Beale function 정의
def beale(x, y):
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y ** 2) ** 2
    term3 = (2.625 - x + x * y ** 3) ** 2
    return term1 + term2 + term3

# 초기값 설정
x = Variable(np.array(1.0))
y = Variable(np.array(1.0))

# Beale 함수 적용 및 역전파
z = beale(x, y)
z.backward()

# gradient 출력
print(f"x.grad = {x.grad}, y.grad = {y.grad}")
