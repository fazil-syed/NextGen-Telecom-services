import sqlite3


conn = sqlite3.connect('user_auth.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER UNIQUE,
    email TEXT UNIQUE,
    password TEXT
)''')

conn.commit()
conn.close()