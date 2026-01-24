from flask import Flask, request, make_response, redirect
from markupsafe import escape

app = Flask(__name__)
app.secret_key = "hackathon_secret"

# == MOCK DATABASE ==
USERS = {
    "1": {"first": "admin", "last": "admin", "user": "admin", "pass": "password"},
    "2": {"first": "Gordon", "last": "Brown", "user": "gordonb", "pass": "abc12345"},
}

@app.route('/')
def index():
    if not request.cookies.get('PHPSESSID'):
        return redirect('/login.php')
    return "<h1>Welcome to Damn Vulnerable Web Application</h1>"

@app.route('/login.php', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        resp = make_response(redirect('/'))
        resp.set_cookie('PHPSESSID', 'mock_session_id')
        resp.set_cookie('security', 'low')
        return resp
    return "<form method='post'><input name='username'><input name='password'><input type='submit'></form>"

@app.route('/vulnerabilities/sqli/')
def sqli():
    # Check auth
    if request.cookies.get('security') != 'low': return redirect('/login.php')

    user_id = request.args.get('id', '')
    response = f"User ID: {escape(user_id)}<br>"

    # PATCHED: Fixed SQL Injection (UNION-based) by removing vulnerable keyword detection that leaked data and using strict numeric input validation (int conversion) to ensure only valid numeric IDs are looked up in the mock DB. Also escaped all user-controlled output to prevent XSS.
    try:
        uid = str(int(user_id))
        if uid in USERS:
            u = USERS[uid]
            response += f"ID: {escape(uid)}<br>First: {escape(u['first'])}<br>Surname: {escape(u['last'])}"
    except ValueError:
        pass  # Invalid ID format, no user info shown

    return response

if __name__ == '__main__':
    app.run(port=80, debug=True) # Runs on port 80 to match real DVWA