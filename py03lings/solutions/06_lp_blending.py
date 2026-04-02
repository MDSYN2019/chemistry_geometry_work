"""Solution 06: linear programming blending model with PuLP."""

import pulp


PROFIT_REGULAR = 30
PROFIT_DELUXE = 50


def build_and_solve() -> tuple[float, float, float]:
    problem = pulp.LpProblem("blending", pulp.LpMaximize)

    regular = pulp.LpVariable("regular", lowBound=0)
    deluxe = pulp.LpVariable("deluxe", lowBound=0)

    problem += PROFIT_REGULAR * regular + PROFIT_DELUXE * deluxe

    problem += regular + 2 * deluxe <= 40
    problem += 2 * regular + deluxe <= 50

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return (
        float(regular.value()),
        float(deluxe.value()),
        float(pulp.value(problem.objective)),
    )


if __name__ == "__main__":
    regular, deluxe, profit = build_and_solve()
    print(f"regular={regular:.2f}, deluxe={deluxe:.2f}, profit={profit:.2f}")
