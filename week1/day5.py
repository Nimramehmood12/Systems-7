from datetime import datetime

class BankAccount:
    def __init__(self, account_number, account_holder, initial_balance=0.0):
        if initial_balance < 0: raise ValueError('Initial balance cannot be negative.')
        self.account_number, self.account_holder = account_number, account_holder
        self._balance, self._ledger = float(initial_balance), []
        self._record('Account Opened', self._balance)
    @property
    def balance(self): return self._balance
    def _record(self, kind, amount):
        self._ledger.append({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': kind, 'amount': amount, 'balance': self._balance})
    def deposit(self, amount):
        if amount <= 0: print('Deposit must be greater than zero.'); return False
        self._balance += amount; self._record('Deposit', amount); print(f'Deposited ${amount:.2f}. New Balance: ${self._balance:.2f}'); return True
    def withdraw(self, amount):
        if amount <= 0 or amount > self._balance: print('Withdrawal denied.'); return False
        self._balance -= amount; self._record('Withdrawal', amount); print(f'Withdrew ${amount:.2f}. Remaining Balance: ${self._balance:.2f}'); return True
    def statement(self):
        for item in self._ledger: print(item)

def main():
    account = BankAccount('PK-9921', 'Nimra M.', 500)
    account.deposit(250); account.withdraw(120); account.withdraw(900); account.statement()

if __name__ == '__main__': main()
