import json, os
FILE_NAME = 'student_records.json'

def load_records(filename=FILE_NAME):
    if not os.path.exists(filename): return {}
    try:
        with open(filename, encoding='utf-8') as file: return json.load(file)
    except (json.JSONDecodeError, OSError): return {}

def save_records(records, filename=FILE_NAME):
    with open(filename, 'w', encoding='utf-8') as file: json.dump(records, file, indent=4)
    print('Records saved successfully.')

def main():
    records = load_records()
    while True:
        print('\n1. View All Students\n2. Add/Update Student\n3. Exit')
        choice = input('Enter choice: ').strip()
        if choice == '1':
            if not records: print('No student records found.')
            for sid, info in records.items(): print(sid, info)
        elif choice == '2':
            sid = input('Student ID: ').strip(); name = input('Name: ').strip()
            try:
                gpa = float(input('GPA (0.0-4.0): '))
                if not 0 <= gpa <= 4: raise ValueError
            except ValueError: print('Invalid GPA.'); continue
            major = input('Major: ').strip(); records[sid] = {'name': name, 'gpa': gpa, 'major': major}; save_records(records)
        elif choice == '3': break
        else: print('Invalid option.')

if __name__ == '__main__': main()
