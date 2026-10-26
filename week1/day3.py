import random

def main():
    target = random.randint(1, 100)
    attempts = 0
    print('A random number between 1 and 100 has been chosen.')
    while True:
        value = input("Enter your guess or 'q' to quit: ").strip()
        if value.lower() == 'q':
            print(f'Game aborted. Target was {target}.'); break
        try: guess = int(value)
        except ValueError: print('Please enter a valid integer.'); continue
        attempts += 1
        if guess < target: print('Too low!')
        elif guess > target: print('Too high!')
        else: print(f'Correct! Found {target} in {attempts} attempts.'); break

if __name__ == '__main__': main()
