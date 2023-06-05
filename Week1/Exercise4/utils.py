
##############################################################################
# TODO: Initialize test cases.
# 1) x^2 + 1
# 2) x^2 + 2x + 1
# 3) 5x^2 + 6x + 10
##############################################################################
def regconize_formula(f: str) -> dict:
    dic, pattern = {}, []
    f = f.split(' ')
    i = 0
    while i < len(f):
        if f[i] == "-":
            pattern.append("-" + f[i + 1])
            i += 1
        elif f[i] != "+":
            pattern.append(f[i])
        i += 1

    for s in pattern:
        if "x^4" in s:
            dic["x^4"] = float(s[:s.index("x")])   
        elif "x^3" in s:
            dic["x^3"] = float(s[:s.index("x")])
        elif "x^2" in s:
            dic["x^2"] = float(s[:s.index("x")])
        elif "x" in s:
            dic["x"] = float(s[:s.index("x")])
        else:
            dic["const"] = float(s)
    return dic

def derivative(f, x):
    dfdx = 0
    dic = regconize_formula(f)
    for key, value in dic.items():
        if key == "x^4":
            dfdx += 4 * value * x**3
        elif key == "x^3":
            dfdx += 3 * value * x**2
        elif key == "x^2":
            dfdx += 2 * value * x
        elif key == "x":
            dfdx += value
    return dfdx

def invertHessian(f, x):
    dfdxdx = 0
    dic = regconize_formula(f)
    for key, value in dic.items():
        if key == "x^4":
            dfdxdx += 12 * value * x**2
        elif key == "x^3":
            dfdxdx += 6 * value * x 
        elif key == "x^2":
            dfdxdx += 2 * value
    return 1 / dfdxdx