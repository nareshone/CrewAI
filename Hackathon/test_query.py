import sqlite3

# ✅ Connect to SQLite database
conn = sqlite3.connect("sample.db")  # Change to your DB file path
cursor = conn.cursor()

# ✅ Write your SQL query
query = "SELECT employee_id, first_name, last_name, department, UPPER(department) AS department_uppercase FROM employees LIMIT 50"  # Example query

# ✅ Execute the query
cursor.execute(query)

# ✅ Fetch results
results = cursor.fetchall()

# ✅ Print results
for row in results:
    print(row)

# ✅ Close connection
conn.close()