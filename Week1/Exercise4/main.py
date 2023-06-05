import argparse
import random
from utils import derivative
from utils import invertHessian
##############################################################################
# TODO: Initialize initial state.
# 1) initial value of x                                   
# 2) learning_rate: eta
# 3) tolerance: threshold to stop loop
##############################################################################
cnt = 0
def gradient_descent(f):
    global cnt
    x = random.random()
    learning_rate = 0.01
    if args.useHessian == True:
        learning_rate = invertHessian(f, x)
    tolerance = 0.00001 
    
    while True:
        cnt += 1
        dfdx = derivative(f, x)
        if abs(dfdx) < tolerance:
            break 
        x = x - learning_rate * dfdx
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--useHessian", type=bool, default=False, 
                        help="Gradient use inverted Hessian as learning rate")
    args = parser.parse_args()

    f = "10x^4 + 5x^3 + 10x^2 - 6x + 10"
    print(gradient_descent(f))
    print("# of Iterations: ", cnt)