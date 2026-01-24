import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username text, password text)''')
    c.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'password')")
    c.execute("INSERT OR IGNORE INTO users VALUES ('user', 'pass')")
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Login</h2>
        <form method="post" action="/login">
            Username: <input name="username"><br><br>
            Password: <input name="password" type="password"><br><br>
            <input type="submit" value="Login">
        </form>
    </body>
    </html>
    '''

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # VULNERABLE TO SQL INJECTION
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    print(f"Executed query: {query}")  # For debugging
    c.execute(query)
    user = c.fetchone()
    conn.close()
    
    if user:
        return f"Logged in successfully as {user[0]}!"
    else:
        return "Invalid credentials. Try: admin / password or SQL injection like admin'--"

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)