import sqlite3
import pandas as pd
import os

db_path = 'saltguard.db'

# Connect to database
conn = sqlite3.connect(db_path)

print("Converting patients.csv -> patients table...")
df_patients = pd.read_csv('patients.csv')
df_patients.to_sql('patients', conn, if_exists='replace', index=False)

print("Converting admissions.csv -> admissions table...")
df_admissions = pd.read_csv('admissions.csv')
df_admissions.to_sql('admissions', conn, if_exists='replace', index=False)

print("Converting D_LABITEMS.csv -> d_labitems table...")
df_d_lab = pd.read_csv('d_labitems.csv')
df_d_lab.to_sql('d_labitems', conn, if_exists='replace', index=False)

print("Converting final.csv -> final_features table...")
df_final = pd.read_csv('final.csv')
df_final.to_sql('final_features', conn, if_exists='replace', index=False)

# Lab events are 11 MB. SQLite can digest this easily.
print("Converting labevents.csv -> labevents table...")
df_lab = pd.read_csv('labevents.csv')
df_lab.to_sql('labevents', conn, if_exists='replace', index=False)

# Create some fast indexes
cur = conn.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_subject_id ON patients(subject_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON admissions(subject_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_admissions_hadm_id ON admissions(hadm_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_labevents_hadm_id ON labevents(hadm_id);")
conn.commit()

conn.close()
print("Success! Created lightweight 'saltguard.db' SQLite Database.")
