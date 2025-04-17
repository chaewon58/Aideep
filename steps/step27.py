import numpy as np
import math
from dezero import Variable, Function
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.001):
    y = Variable(np.array(0.0))  # 초기값을 0으로 설정
    term = Variable(np.array(0.0))  # 각 항을 추적하기 위한 변수
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        term.data = t.data  # 현재 항을 저장
        y = y + term
        if abs(term.data) < threshold:
            break
    return y


# x 값 설정 (x = 3파이/4)
x = Variable(np.array(3 * np.pi / 4))

# my_sin 계산
y = my_sin(x, threshold=0.001)
y.backward()

# 결과 출력
print('--- approximate sin ---')
print(f'sin({x.data}) = {y.data}')
print(f'Gradient: {x.grad}')

# 계산 그래프 출력
x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin_graph.png')
