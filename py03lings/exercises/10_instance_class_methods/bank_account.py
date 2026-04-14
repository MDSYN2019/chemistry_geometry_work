"""Exercise 10: practice instance methods and class methods."""


class BankAccount:
    bank_name = "Py03 Credit Union"

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> float:
        # TODO: raise ValueError if amount <= 0
        # TODO: add amount to balance
        # TODO: return updated balance
        return self.balance

    def withdraw(self, amount: float) -> float:
        # TODO: raise ValueError if amount <= 0
        # TODO: raise ValueError if amount > balance
        # TODO: subtract amount and return updated balance
        return self.balance

    @classmethod
    def starter_account(cls, owner: str) -> "BankAccount":
        # TODO: return an account with a 25.0 starter bonus
        return cls(owner, 0.0)


if __name__ == "__main__":
    account = BankAccount.starter_account("Avery")
    account.deposit(10)
    account.withdraw(5)
    print(account.owner, account.balance, account.bank_name)
