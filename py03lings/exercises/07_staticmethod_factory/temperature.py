"""Exercise 07: use staticmethod for unit conversion and parsing."""


class Temperature:
    def __init__(self, celsius: float) -> None:
        self.celsius = celsius

    @staticmethod
    def from_fahrenheit(fahrenheit: float) -> "Temperature":
        # TODO: convert Fahrenheit to Celsius
        # TODO: return Temperature(celsius_value)
        return Temperature(0.0)

    @staticmethod
    def is_reasonable_celsius(value: float) -> bool:
        # TODO: return True only for values in [-100, 100]
        return False


if __name__ == "__main__":
    t = Temperature.from_fahrenheit(77)
    print(t.celsius)
    print(Temperature.is_reasonable_celsius(t.celsius))
