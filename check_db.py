import sqlite3
import sys
import os

def print_table(cursor, table_name):
    print(f"\n=== {table_name} ===")
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        if rows:
            # Print column names
            columns = [description[0] for description in cursor.description]
            print("Columns:", columns)
            # Print rows
            for row in rows:
                print(row)
        else:
            print("No rows found")
    except sqlite3.Error as e:
        print(f"Error reading table {table_name}: {e}")

# Print current directory and check if database exists
print("Current directory:", os.getcwd())
db_path = 'instance/app.db'
print("Looking for database at:", os.path.abspath(db_path))
print("Database exists:", os.path.exists(db_path))

try:
    # Connect to database
    print("\nTrying to connect to database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List all tables
    print("\nQuerying tables...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("Available tables:", [table[0] for table in tables])
    
    # Print contents of important tables
    important_tables = ['organizations', 'clients', 'projects', 'project_clients']
    for table in important_tables:
        if (table,) in tables:  # Check if table exists
            print_table(cursor, table)
        else:
            print(f"\nTable '{table}' does not exist")

except sqlite3.Error as e:
    print("SQLite error:", e)
    print("Error type:", type(e).__name__)
except Exception as e:
    print("General error:", e)
    print("Error type:", type(e).__name__)
finally:
    if 'conn' in locals():
        conn.close() 