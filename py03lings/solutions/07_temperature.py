"""Solution 07: use staticmethod for unit conversion and parsing."""


class Temperature:
    def __init__(self, celsius: float) -> None:
        self.celsius = celsius

    @staticmethod
    def from_fahrenheit(fahrenheit: float) -> "Temperature":
        celsius_value = (fahrenheit - 32) * 5 / 9
        return Temperature(celsius_value)

    @staticmethod
    def is_reasonable_celsius(value: float) -> bool:
        return -100 <= value <= 100


if __name__ == "__main__":
    t = Temperature.from_fahrenheit(77)
    print(t.celsius)
    print(Temperature.is_reasonable_celsius(t.celsius))
