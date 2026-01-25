#!/usr/bin/env python3
"""
SkyHammer SQLi Factory
Generates synthetic SQL injection vulnerabilities for benchmarking and RLAIF training.
"""

import os
import random
import json
import argparse
from datetime import datetime


# Randomization pools
TABLE_NAMES = ["users", "staff", "employees", "accounts", "customers", "admins", "members", "profiles"]
PARAM_NAMES = ["id", "user", "username", "q", "search", "name", "email", "query", "term", "input"]
COLUMN_NAMES = ["name", "username", "email", "password", "id", "user_id", "login", "account"]
ROUTE_NAMES = ["/search", "/login", "/user", "/query", "/find", "/lookup", "/get", "/fetch", "/api/user"]
DB_NAMES = ["app.db", "database.db", "db.sqlite", "data.db", "users.db", "main.db"]


def generate_flask_fstring(run_id: int) -> dict:
    """Flask app with f-string SQL injection"""
    table = random.choice(TABLE_NAMES)
    param = random.choice(PARAM_NAMES)
    column = random.choice(COLUMN_NAMES)
    route = random.choice(ROUTE_NAMES)
    db = random.choice(DB_NAMES)

    code = f'''import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{route}')
def search():
    {param} = request.args.get('{param}')
    # VULNERABILITY: Direct f-string interpolation in SQL query
    query = f"SELECT * FROM {table} WHERE {column} = '{{ {param} }}'"
    conn = sqlite3.connect('{db}')
    cursor = conn.cursor()
    cursor.execute(query)  # <--- SQL INJECTION HERE
    results = cursor.fetchall()
    conn.close()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

    secure_code = f'''import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{route}')
def search():
    {param} = request.args.get('{param}')
    # SECURE: Parameterized query with placeholder
    query = "SELECT * FROM {table} WHERE {column} = ?"
    conn = sqlite3.connect('{db}')
    cursor = conn.cursor()
    cursor.execute(query, ({param},))  # <--- PARAMETERIZED
    results = cursor.fetchall()
    conn.close()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

    exploit = f"curl '{route}?{param}=' OR '1'='1' --"

    return {
        "id": f"sqli_flask_fstring_{run_id:03d}",
        "type": "sqli",
        "subtype": "flask_fstring",
        "vulnerable_code": code,
        "secure_code": secure_code,
        "exploit_command": exploit,
        "param_name": param,
        "route": route
    }


def generate_flask_format(run_id: int) -> dict:
    """Flask app with .format() SQL injection"""
    table = random.choice(TABLE_NAMES)
    param = random.choice(PARAM_NAMES)
    column = random.choice(COLUMN_NAMES)
    route = random.choice(ROUTE_NAMES)
    db = random.choice(DB_NAMES)

    code = f'''import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{route}')
def get_user():
    {param} = request.args.get('{param}', '')
    # VULNERABILITY: String format in SQL query
    query = "SELECT * FROM {table} WHERE {column} = '{{}}'".format({param})
    conn = sqlite3.connect('{db}')
    cursor = conn.cursor()
    cursor.execute(query)  # <--- SQL INJECTION HERE
    results = cursor.fetchall()
    conn.close()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

    secure_code = f'''import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{route}')
def get_user():
    {param} = request.args.get('{param}', '')
    # SECURE: Parameterized query
    query = "SELECT * FROM {table} WHERE {column} = ?"
    conn = sqlite3.connect('{db}')
    cursor = conn.cursor()
    cursor.execute(query, ({param},))  # <--- PARAMETERIZED
    results = cursor.fetchall()
    conn.close()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

    exploit = f"curl '{route}?{param}=admin'--"

    return {
        "id": f"sqli_flask_format_{run_id:03d}",
        "type": "sqli",
        "subtype": "flask_format",
        "vulnerable_code": code,
        "secure_code": secure_code,
        "exploit_command": exploit,
        "param_name": param,
        "route": route
    }


def generate_flask_concat(run_id: int) -> dict:
    """Flask app with string concatenation SQL injection"""
    table = random.choice(TABLE_NAMES)
    param = random.choice(PARAM_NAMES)
    column = random.choice(COLUMN_NAMES)
    route = random.choice(ROUTE_NAMES)
    db = random.choice(DB_NAMES)

    code = f'''import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{route}', methods=['POST'])
def authenticate():
    {param} = request.form.get('{param}')
    password = request.form.get('password')
    # VULNERABILITY: String concatenation in SQL
    query = "SELECT * FROM {table} WHERE {column} = '" + {param} + "' AND password = '" + password + "'"
    conn = sqlite3.connect('{db}')
    cursor = conn.cursor()
    cursor.execute(query)  # <--- SQL INJECTION HERE
    user = cursor.fetchone()
    conn.close()
    if user:
        return jsonify({{"status": "authenticated"}})
    return jsonify({{"status": "failed"}}), 401

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

    secure_code = f'''import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('{route}', methods=['POST'])
def authenticate():
    {param} = request.form.get('{param}')
    password = request.form.get('password')
    # SECURE: Parameterized query
    query = "SELECT * FROM {table} WHERE {column} = ? AND password = ?"
    conn = sqlite3.connect('{db}')
    cursor = conn.cursor()
    cursor.execute(query, ({param}, password))  # <--- PARAMETERIZED
    user = cursor.fetchone()
    conn.close()
    if user:
        return jsonify({{"status": "authenticated"}})
    return jsonify({{"status": "failed"}}), 401

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

    exploit = f"curl -X POST '{route}' -d '{param}=admin' OR '1'='1'--&password=x'"

    return {
        "id": f"sqli_flask_concat_{run_id:03d}",
        "type": "sqli",
        "subtype": "flask_concat",
        "vulnerable_code": code,
        "secure_code": secure_code,
        "exploit_command": exploit,
        "param_name": param,
        "route": route
    }


def generate_fastapi_fstring(run_id: int) -> dict:
    """FastAPI app with f-string SQL injection"""
    table = random.choice(TABLE_NAMES)
    param = random.choice(PARAM_NAMES)
    column = random.choice(COLUMN_NAMES)
    route = random.choice(ROUTE_NAMES)
    db = random.choice(DB_NAMES)

    code = f'''import sqlite3
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("{route}")
async def search({param}: str = Query(...)):
    # VULNERABILITY: f-string in SQL
    query = f"SELECT * FROM {table} WHERE {column} = '{{{param}}}'"
    conn = sqlite3.connect("{db}")
    cursor = conn.cursor()
    cursor.execute(query)  # <--- SQL INJECTION HERE
    results = cursor.fetchall()
    conn.close()
    return {{"results": results}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    secure_code = f'''import sqlite3
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("{route}")
async def search({param}: str = Query(...)):
    # SECURE: Parameterized query
    query = "SELECT * FROM {table} WHERE {column} = ?"
    conn = sqlite3.connect("{db}")
    cursor = conn.cursor()
    cursor.execute(query, ({param},))  # <--- PARAMETERIZED
    results = cursor.fetchall()
    conn.close()
    return {{"results": results}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    exploit = f"curl '{route}?{param}=' OR '1'='1'--"

    return {
        "id": f"sqli_fastapi_fstring_{run_id:03d}",
        "type": "sqli",
        "subtype": "fastapi_fstring",
        "vulnerable_code": code,
        "secure_code": secure_code,
        "exploit_command": exploit,
        "param_name": param,
        "route": route
    }


def generate_raw_execute(run_id: int) -> dict:
    """Raw cursor.execute with string injection"""
    table = random.choice(TABLE_NAMES)
    param = random.choice(PARAM_NAMES)
    column = random.choice(COLUMN_NAMES)
    db = random.choice(DB_NAMES)

    code = f'''import sqlite3

def get_user({param}):
    """Get user from database"""
    conn = sqlite3.connect("{db}")
    cursor = conn.cursor()
    # VULNERABILITY: String interpolation in execute
    cursor.execute("SELECT * FROM {table} WHERE {column} = '%s'" % {param})
    result = cursor.fetchone()
    conn.close()
    return result

# Example usage
if __name__ == "__main__":
    user_input = input("Enter {param}: ")
    print(get_user(user_input))
'''

    secure_code = f'''import sqlite3

def get_user({param}):
    """Get user from database"""
    conn = sqlite3.connect("{db}")
    cursor = conn.cursor()
    # SECURE: Parameterized query
    cursor.execute("SELECT * FROM {table} WHERE {column} = ?", ({param},))
    result = cursor.fetchone()
    conn.close()
    return result

# Example usage
if __name__ == "__main__":
    user_input = input("Enter {param}: ")
    print(get_user(user_input))
'''

    exploit = f"echo \"' OR '1'='1\" | python script.py"

    return {
        "id": f"sqli_raw_execute_{run_id:03d}",
        "type": "sqli",
        "subtype": "raw_execute",
        "vulnerable_code": code,
        "secure_code": secure_code,
        "exploit_command": exploit,
        "param_name": param,
        "route": None
    }


# Generator functions pool
GENERATORS = [
    generate_flask_fstring,
    generate_flask_format,
    generate_flask_concat,
    generate_fastapi_fstring,
    generate_raw_execute,
]


def generate_benchmark_set(n_total: int = 50, n_test: int = 10, output_dir: str = "data/benchmark_set"):
    """Generate the full benchmark set with train/test split"""
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    for i in range(n_total):
        generator = random.choice(GENERATORS)
        sample = generator(i)
        sample["generated_at"] = datetime.now().isoformat()
        all_samples.append(sample)

        # Also save individual file
        filepath = os.path.join(output_dir, f"{sample['id']}.json")
        with open(filepath, "w") as f:
            json.dump(sample, f, indent=2)

    # Split into train/test
    random.shuffle(all_samples)
    test_set = all_samples[:n_test]
    train_set = all_samples[n_test:]

    # Save splits
    with open(os.path.join(output_dir, "test_set.json"), "w") as f:
        json.dump(test_set, f, indent=2)

    with open(os.path.join(output_dir, "train_set.json"), "w") as f:
        json.dump(train_set, f, indent=2)

    # Save full set
    with open(os.path.join(output_dir, "full_set.json"), "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"Generated {n_total} SQLi benchmark samples:")
    print(f"  - Train set: {len(train_set)} samples")
    print(f"  - Test set: {len(test_set)} samples")
    print(f"  - Output directory: {output_dir}")

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="SQLi Benchmark Factory")
    parser.add_argument("--n_total", type=int, default=50, help="Total samples to generate")
    parser.add_argument("--n_test", type=int, default=10, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="data/benchmark_set", help="Output directory")

    args = parser.parse_args()

    generate_benchmark_set(args.n_total, args.n_test, args.output_dir)


if __name__ == "__main__":
    main()
