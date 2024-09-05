import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('user_auth.db')
c = conn.cursor()

# Add an admin user
admin_email = 'saujanya@gmail.com'
admin_password = 'saujanya'

# Check if the admin user already exists
c.execute('SELECT * FROM users WHERE email = ?', (admin_email,))
if not c.fetchone():
    c.execute('INSERT INTO users (customer_id, email, password) VALUES (?, ?, ?)', 
              (None, admin_email, admin_password))  # Use None for customer_id as it is auto-incremented
    conn.commit()

conn.close()