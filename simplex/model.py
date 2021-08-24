import numpy as np

class Model:
    """Model for linear programming defined by:
    max c^T x
    s.t. Ax <= b
    x >= 0
    """

    def __init__(self):
        self.tableau: np.ndarray = np.array([], np.longdouble)
        self.constraints = []
        self.b = []

    def set_objective(self, c: np.ndarray):
        self.c = -c

    def add_constraint(self, constraint: np.ndarray, b: np.longdouble):
        self.constraints.append(constraint)
        self.b.append(b)

    def _create_tableau(self):
        self.tableau = np.zeros(
            (len(self.constraints) + 1, len(self.c) + 2 * len(self.constraints) + 1),
            np.longdouble,
        )

        line = np.hstack(
            [[0] * len(self.constraints), self.c, [0] * (len(self.constraints) + 1)]
        ).astype(np.longdouble)

        self.tableau[0, :] = line
        for i, (constraint, b) in enumerate(zip(self.constraints, self.b)):
            unit = [0] * len(self.constraints)
            unit[i] = 1
            line = np.hstack([unit, constraint, unit, b])
            self.tableau[i + 1] = line

    def _pivot(self, i, j, c):
        self.tableau[i] = self.tableau[i] / self.tableau[i, j]
        for idx, line in enumerate(self.tableau):

            if idx != i:
                if idx == 0 and c is not None:
                    c = c - self.tableau[i] * line[j]

                self.tableau[idx] = line - self.tableau[i] * line[j]

    def _solve_aux(self):
        c = np.concatenate(
            (
                self.tableau[0, 0:-1].copy(),
                [0] * len(self.constraints),
                [self.tableau[0, -1]],
            )
        ).astype(np.longdouble)
        for i, line in enumerate(self.tableau[1:]):
            if line[-1] < 0:
                self.tableau[i + 1] = -line
        self.tableau[0, :-1] = self.tableau[0, :-1] * 0
        new_vars = np.vstack(
            ([1] * len(self.constraints), np.identity(len(self.constraints)))
        ).astype(np.longdouble)
        lastcol = self.tableau[:, -1]
        self.tableau = np.concatenate(
            (self.tableau[:, :-1], new_vars, lastcol.reshape((*lastcol.shape, 1))),
            axis=1,
        )

        for line in self.tableau[1:]:
            self.tableau[0] = self.tableau[0] - line

        sol = self.solve(create_tableau=False, c=c)

        self.tableau = np.delete(
            self.tableau, np.s_[-len(self.constraints) - 1 : -1], axis=1
        )
        c = np.delete(c, np.s_[-len(self.constraints) - 1 : -1])
        opt = round(sol[1], 8)
        # (optimal solution, variables, certificate, first line)
        return opt, sol[2], sol[3], c

    def solve(self, print_tableau=False, create_tableau=True, c=None):
        if create_tableau:
            self._create_tableau()
            self.aux_sol = self._solve_aux()
            if np.all(self.aux_sol[0] < 0):
                # Inviavel
                return ("inviavel", self.aux_sol[2])
            self.tableau[0] = self.aux_sol[3]

        if print_tableau:
            print(self.tableau)

        self.tableau = np.around(self.tableau, 12)
        neg_c = np.where(self.tableau[0, len(self.constraints) : -1] < 0)

        while neg_c[0].size:
            neg_c = len(self.constraints) + neg_c[0][0]

            if np.isclose(self.tableau[0, neg_c], 0):
                self.tableau[0, neg_c] = 0
                neg_c = np.where(self.tableau[0, len(self.constraints) : -1] < 0)
                continue

            b = self.tableau[1:, -1]
            col = self.tableau[1:, neg_c]

            if np.all(col <= 0):
                # Ilimitada
                d = np.array([0] * (len(self.constraints)), np.longdouble)
                for i in range(len(self.constraints), len(self.constraints) + len(d)):
                    if i != neg_c:
                        col = self.tableau[:, i]
                        if (
                            np.count_nonzero(col == 1) == 1
                            and np.count_nonzero(col == 0) == len(col) - 1
                        ):
                            d[i - len(self.constraints)] = -self.tableau[
                                np.argmax(col), neg_c
                            ]
                d[neg_c - len(self.constraints)] = 1

                return ("ilimitada", self.aux_sol[1], d[:len(self.c)])


            ratios = b / col
            min_ratios = np.where(ratios == np.amin(ratios[col > 0]))[0]
            pivot = (min_ratios[0] + 1, neg_c)
            if ratios[min_ratios[0]] == 0:
                # Fiz random, não vou implementar Bland, que eu preciso acabar isso logo
                pivot = (min_ratios[random.randint(0, len(min_ratios) - 1)] + 1, neg_c)
            self._pivot(*pivot, c)

            if print_tableau:
                print(self.tableau)

            self.tableau = np.around(self.tableau, 12)
            neg_c = np.where(self.tableau[0, len(self.constraints) : -1] < 0)

        x = np.array([0] * len(self.c), np.longdouble)
        for i in range(len(self.constraints), len(self.constraints) + len(self.c)):
            col = self.tableau[:, i]
            if (
                np.count_nonzero(col == 1) == 1
                and np.count_nonzero(col == 0) == len(col) - 1
            ):
                x[i - len(self.constraints)] = self.tableau[np.argmax(col), -1]
        # (status, objective value, variables, certificate)
        return (
            "otima",
            self.tableau[0, -1],
            x,
            self.tableau[0, 0 : len(self.constraints)],
        )
