"""Exercise 06: linear programming blending model with PuLP.

Goal: maximize profit for a 2-product production plan.

Install once:
    pip install pulp
"""

import pulp


# Profit per unit
PROFIT_REGULAR = 30
PROFIT_DELUXE = 50


def build_and_solve() -> tuple[float, float, float]:
    """Return (regular_units, deluxe_units, max_profit)."""
    # TODO: create a maximization problem named "blending"
    problem = None

    # TODO: create non-negative variables regular and deluxe
    regular = None
    deluxe = None

    # TODO: add objective: maximize 30*regular + 50*deluxe

    # Hours limits
    # Mixing: 1*regular + 2*deluxe <= 40
    # Packaging: 2*regular + 1*deluxe <= 50
    # TODO: add both constraints

    # TODO: solve the model

    # TODO: return solution tuple as floats:
    # (regular_value, deluxe_value, objective_value)
    return 0.0, 0.0, 0.0


if __name__ == "__main__":
    regular, deluxe, profit = build_and_solve()
    print(f"regular={regular:.2f}, deluxe={deluxe:.2f}, profit={profit:.2f}")
