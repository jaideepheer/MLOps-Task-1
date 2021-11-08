# MLOps-Task-1

Hi all,
In today's class we will
- ✔️ discuss how did the homework of "model comparison" go.
- ✔️ make a small dummy hello world app using flask, https://flask.palletsprojects.com/en/2.0.x/quickstart/
- ❌ we will expand the flask app to do something with a small logic (given a number, output if it is odd or even)
- ❌ we will expand it further to serve our model
- ❌ and ponder how would we give out "image" input (shall we vectorize and provide or what?)

## How to run

```
python api/app.py
```

## Note

Instead of using [noirbizarre/flask-restplus](https://github.com/noirbizarre/flask-restplus), I am using the newer fork [python-restx/flask-restx](https://github.com/python-restx/flask-restx).

## Run Output

```
* Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 123-170-903
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET /swaggerui/swagger-ui-bundle.js HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET /swaggerui/droid-sans.css HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET /swaggerui/swagger-ui-standalone-preset.js HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET /swaggerui/swagger-ui.css HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET /swagger.json HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:31:48] "GET /swaggerui/favicon-32x32.png HTTP/1.1" 200 -
127.0.0.1 - - [08/Nov/2021 19:32:04] "GET /hello HTTP/1.1" 200 -
```