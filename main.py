import numpy as np
import re
from simplex import Model
from utils import print_solution

if __name__ == "__main__":
    np.seterr(divide='ignore')
    nconstraints, _ = map(int, re.findall(r"\d+", input()))
    c = np.array(list(map(np.longdouble, re.findall(r"-?\d+", input()))), np.longdouble)
    model = Model()
    model.set_objective(c)
    for _ in range(nconstraints):
        constraint = np.array(
            list(map(np.longdouble, re.findall(r"-?\d+", input()))), np.longdouble
        )
        model.add_constraint(constraint[:-1], constraint[-1])

    sol = model.solve()
    print_solution(sol)