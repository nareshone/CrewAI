import os
import sqlite3
import pandas as pd

# === CONFIGURATION ===
csv_folder = "C:\CrewAI\Datasets\hospital"               
db_path = "HOSPITAL.db"       # Output SQLite database

# ✅ Create or connect to SQLite database
conn = sqlite3.connect(db_path)
print(f"✅ Created database: {db_path}")

# ✅ Get list of up to 9 CSV files
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")][:9]

if not csv_files:
    print("⚠️ No CSV files found in folder:", csv_folder)
else:
    for file in csv_files:
        file_path = os.path.join(csv_folder, file)
        table_name = os.path.splitext(file)[0].lower().replace(" ", "_")  # safe table name

        try:
            df = pd.read_csv(file_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"📥 Loaded '{file}' into table '{table_name}'")
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

conn.close()
print("✅ Done: All CSV files loaded into database.")
