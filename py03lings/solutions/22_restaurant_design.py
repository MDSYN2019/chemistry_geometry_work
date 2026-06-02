"""Solution 22: practice object creation, inheritance, composition, and ABCs."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Dish(ABC):
    def __init__(self, name: str, base_price: float) -> None:
        self.name = name
        self.base_price = float(base_price)

    @classmethod
    def from_menu_row(cls, row: str) -> "Dish":
        name, raw_price = row.split("|", maxsplit=1)
        return cls(name.strip(), float(raw_price))

    @abstractmethod
    def prep_time_minutes(self) -> int:
        """Return the estimated preparation time."""

    def price(self) -> float:
        return self.base_price


class Nigiri(Dish):
    def prep_time_minutes(self) -> int:
        return 4


class MakiRoll(Dish):
    def prep_time_minutes(self) -> int:
        return 6


class Chef:
    def __init__(self, name: str) -> None:
        self.name = name


class Restaurant:
    def __init__(self, chef: Chef, dishes: list[Dish] | None = None) -> None:
        self.chef = chef
        self.dishes = list(dishes or [])

    def add_dish(self, dish: Dish) -> None:
        self.dishes.append(dish)

    def total_prep_time(self) -> int:
        return sum(dish.prep_time_minutes() for dish in self.dishes)

    def menu_names(self) -> list[str]:
        return [dish.name for dish in self.dishes]


if __name__ == "__main__":
    restaurant = Restaurant(Chef("Stephen"))
    restaurant.add_dish(Nigiri.from_menu_row("salmon nigiri|4.5"))
    restaurant.add_dish(MakiRoll.from_menu_row("avocado maki|3.5"))
    print(restaurant.chef.name, restaurant.menu_names(), restaurant.total_prep_time())
