import random
from utils import derivative
##############################################################################
# TODO: Initialize initial state.
# 1) initial value of x                                   
# 2) learning_rate: eta
# 3) tolerance: threshold to stop loop
##############################################################################
def gradient_descent(f):
    x = random.random()
    learning_rate = 0.01
    tolerance = 0.00001 
    
    while True:
        dfdx = derivative(f, x)
        if abs(dfdx) < tolerance:
            break 
        x = x + learning_rate * dfdx
    return x

if __name__ == "__main__":
    f = "-10x^2 - 1000x + 10"
    print(gradient_descent(f))