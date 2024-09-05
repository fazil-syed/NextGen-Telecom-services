# utils/auth.py

import sqlite3

def create_user(customer_id, email, password):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (customer_id, email, password) VALUES (?, ?, ?)', 
              (customer_id, email, password))
    conn.commit()
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    c.execute('SELECT customer_id FROM users WHERE email = ? AND password = ?', 
              (email, password))
    result = c.fetchone()
    conn.close()
    return result

def get_customer_id_by_email(email):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    c.execute('SELECT customer_id FROM users WHERE email = ?', (email,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None
