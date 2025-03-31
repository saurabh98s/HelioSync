import os

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            print(f"Found {name} at: {os.path.join(root, name)}")
            return os.path.join(root, name)
    return None

# Look for database.db in current directory and subdirectories
db_path = find_file('database.db', '.')
if not db_path:
    print("Could not find database.db") 