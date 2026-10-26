def calculator():
    while True:
        print('\n1 Add | 2 Subtract | 3 Multiply | 4 Divide | 5 Back')
        choice = input('Select an option: ').strip()
        if choice == '5': break
        if choice not in {'1','2','3','4'}:
            print('Invalid selection.'); continue
        try:
            a = float(input('Enter first number: ')); b = float(input('Enter second number: '))
        except ValueError:
            print('Invalid numeric input.'); continue
        if choice == '1': result = a + b
        elif choice == '2': result = a - b
        elif choice == '3': result = a * b
        elif b == 0: print('Division by zero is undefined.'); continue
        else: result = a / b
        print(f'Result: {result}')

def main():
    while True:
        print('\n1. Calculator\n2. Exit')
        option = input('Choose: ').strip()
        if option == '1': calculator()
        elif option == '2': break
        else: print('Invalid choice.')

if __name__ == '__main__': main()
