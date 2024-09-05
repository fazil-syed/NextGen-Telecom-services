import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('user_auth.db')
c = conn.cursor()

# Create a table for user authentication
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER UNIQUE,
    email TEXT UNIQUE,
    password TEXT
)''')

conn.commit()
conn.close()