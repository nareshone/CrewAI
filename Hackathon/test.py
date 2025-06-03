import sqlite3

# Connect to the database file
conn = sqlite3.connect('sample.db')
cursor = conn.cursor()

# Get list of tables
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#tables = cursor.fetchall()
#print("Tables:", tables)

# Query data from a table (replace 'your_table_name' with actual name)
cursor.execute("SELECT * FROM employees e;")
rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.execute("SELECT * FROM departments d;")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()