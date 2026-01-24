from flask import Flask, request, make_response
import subprocess
import secrets

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <html>
    <body>
    <h1>Welcome</h1>
    <a href="/login.php">Login</a><br>
    <a href="/vulnerabilities/exec/?ip=127.0.0.1">Vulnerable Endpoint</a>
    </body>
    </html>
    '''

@app.route('/login.php', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session_id = secrets.token_hex(16)
        resp = make_response('Login successful! <a href="/vulnerabilities/exec/?ip=127.0.0.1">Go to exec</a>')
        resp.set_cookie('PHPSESSID', session_id)
        resp.set_cookie('security', 'low')
        return resp
    return '''
    <form method="post">
    Username: <input name="user"><br>
    Password: <input name="pass" type="password"><br>
    <input type="submit" value="Login">
    </form>
    '''

@app.route('/vulnerabilities/exec/', methods=['GET', 'POST'])
def exec_vuln():
    ip = request.args.get('ip', '') if request.method == 'GET' else request.form.get('ip', '')
    if not ip:
        return 'No IP provided. Try: <a href="/vulnerabilities/exec/?ip=127.0.0.1">Ping 127.0.0.1</a>'
    try:
        result = subprocess.run(f"ping -c 4 {ip}", shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
    except Exception as e:
        output = str(e)
    return f'<h2>Ping result for IP: {ip}</h2><pre>{output}</pre><br><a href="/vulnerabilities/exec/?ip=127.0.0.1">Ping 127.0.0.1 again</a>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)