"""Exercise 22: practice object creation, inheritance, composition, and ABCs.

Day 3 focus:
- provide named alternate constructors with ``@classmethod``
- use inheritance for a true is-a relationship
- use composition for has-one and has-many relationships
- require subclass behavior with ``ABC`` and ``@abstractmethod``
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Dish(ABC):
    def __init__(self, name: str, base_price: float) -> None:
        self.name = name
        self.base_price = float(base_price)

    @classmethod
    def from_menu_row(cls, row: str) -> "Dish":
        # TODO: parse "name|price" and construct cls(name, price).
        return cls("TODO", 0.0)

    @abstractmethod
    def prep_time_minutes(self) -> int:
        """Return the estimated preparation time."""

    def price(self) -> float:
        return self.base_price


class Nigiri(Dish):
    # TODO: implement prep_time_minutes() and return 4.
    pass


class MakiRoll(Dish):
    # TODO: implement prep_time_minutes() and return 6.
    pass


class Chef:
    def __init__(self, name: str) -> None:
        self.name = name


class Restaurant:
    def __init__(self, chef: Chef, dishes: list[Dish] | None = None) -> None:
        # has-one composition: a restaurant has one chef.
        self.chef = chef
        # has-many composition: a restaurant has many dishes.
        self.dishes = list(dishes or [])

    def add_dish(self, dish: Dish) -> None:
        # TODO: add dish to this restaurant's menu.
        pass

    def total_prep_time(self) -> int:
        # TODO: sum prep time for all dishes.
        return 0

    def menu_names(self) -> list[str]:
        # TODO: return the names of all dishes.
        return []


if __name__ == "__main__":
    restaurant = Restaurant(Chef("Stephen"))
    restaurant.add_dish(Nigiri.from_menu_row("salmon nigiri|4.5"))
    restaurant.add_dish(MakiRoll.from_menu_row("avocado maki|3.5"))
    print(restaurant.chef.name, restaurant.menu_names(), restaurant.total_prep_time())
