"""Solution 10: practice instance methods and class methods."""


class BankAccount:
    bank_name = "Py03 Credit Union"

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        self.balance += amount
        return self.balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        if amount > self.balance:
            raise ValueError("Insufficient funds.")
        self.balance -= amount
        return self.balance

    @classmethod
    def starter_account(cls, owner: str) -> "BankAccount":
        return cls(owner, 25.0)


if __name__ == "__main__":
    account = BankAccount.starter_account("Avery")
    account.deposit(10)
    account.withdraw(5)
    print(account.owner, account.balance, account.bank_name)
