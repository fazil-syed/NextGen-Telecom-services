import sqlite3


conn = sqlite3.connect('user_auth.db')
c = conn.cursor()


admin_email = 'saujanya@gmail.com'
admin_password = 'saujanya'


c.execute('SELECT * FROM users WHERE email = ?', (admin_email,))
if not c.fetchone():
    c.execute('INSERT INTO users (customer_id, email, password) VALUES (?, ?, ?)', 
              (None, admin_email, admin_password))  
    conn.commit()

conn.close()