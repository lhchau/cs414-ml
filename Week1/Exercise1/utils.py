
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
        if "x^2" in s:
            dic["x^2"] = int(s[:s.index("x")])
        elif "x" in s:
            dic["x"] = int(s[:s.index("x")])
        else:
            dic["const"] = int(s)
    return dic

def derivative(f, x):
    dfdx = 0
    dic = regconize_formula(f)
    for key, value in dic.items():
        if key == "x^2":
            dfdx += 2 * value * x
        elif key == "x":
            dfdx += value
    return dfdx