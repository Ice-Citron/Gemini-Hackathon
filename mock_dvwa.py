from flask import Flask, request, make_response, redirect

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
    response = f"User ID: {user_id}<br>"

    # Simulate vulnerability
    if "' OR '1'='1" in user_id or "UNION" in user_id.upper():
        response += "ID: 1 | First: admin | Last: admin<br>ID: 2 | First: Gordon | Last: Brown"
    elif user_id in USERS:
        u = USERS[user_id]
        response += f"ID: {user_id}<br>First: {u['first']}<br>Surname: {u['last']}"

    return response

if __name__ == '__main__':
    app.run(port=80, debug=True) # Runs on port 80 to match real DVWA