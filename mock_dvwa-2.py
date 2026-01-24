from flask import Flask, request, make_response, redirect
import urllib.parse

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
    if request.cookies.get('security') != 'low': return redirect('/login.php')
    
    # Get ID and decode it (handle %20, +, etc.)
    raw_id = request.args.get('id', '')
    decoded_id = urllib.parse.unquote(raw_id).upper() # Normalize to UPPERCASE
    
    response = f"User ID: {raw_id}<br>"
    
    # == LOOSE CHECKING FOR DEMO RELIABILITY ==
    # If it looks like a SQL injection, we give up the goods.
    is_exploit = False
    if "UNION" in decoded_id and "SELECT" in decoded_id:
        is_exploit = True
    if "OR 1=1" in decoded_id or "OR '1'='1" in decoded_id:
        is_exploit = True
    
    if is_exploit:
        response += "<pre>"
        response += "ID: 1 | First: admin | Last: admin | User: admin | Pass: 5f4dcc3b5aa765d61d8327deb882cf99<br>"
        response += "ID: 2 | First: Gordon | Last: Brown | User: gordonb | Pass: e834324523452345<br>"
        response += "</pre>"
    elif raw_id in USERS:
        u = USERS[raw_id]
        response += f"<pre>ID: {raw_id}<br>First: {u['first']}<br>Surname: {u['last']}</pre>"
            
    return response

if __name__ == '__main__':
    # Run on port 80 to match standard HTTP
    app.run(port=80, debug=True)