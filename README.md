# MLOps-Task-1

Hi all,
In today's class we will
- ✔️ discuss how did the homework of "model comparison" go.
- ✔️ make a small dummy hello world app using flask, https://flask.palletsprojects.com/en/2.0.x/quickstart/
- ✔️ we will expand the flask app to do something with a small logic (given a number, output if it is odd or even)
- ✔️ we will expand it further to serve our model
- ✔️ and ponder how would we give out "image" input (shall we vectorize and provide or what?)

## How to run

```
cd api
flask run
```

### Predict

The following `POST` request should be sent to `/predict` on the server.

```
curl http://localhost:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image":
["0.0","0.0","0.0","2.000000000000008","12.99999999999999","2.3092638912203262e-14","0.0","0.0","0.0","0.0","0.0","7.99999999999998","14.999999999999988","2.664535259100375e-14","0.0","0.0","0.0","0.0","4.9999999999999885","15.999999999999975","5.000000000000027","2.0000000000000027","3.552713678800496e-15","0.0","0.0","0.0","14.999999999999975","12.000000000000007","1.0000000000000182","15.999999999999961","4.000000000000018","7.1054273576009955e-15","3.5527136788004978e-15","3.9999999999999925","15.999999999999984","2.0000000000000275","8.999999999999984","15.999999999999988","8.00000000000001","1.4210854715201997e-14","3.1554436208840472e-30","3.5527136788004974e-15","9.999999999999995","13.999999999999986","15.99999999999999","16.0","4.000000000000025","7.105427357601008e-15","0.0","0.0","0.0","0.0","12.999999999999982","8.000000000000009","1.4210854715202004e-14","0.0","0.0","0.0","0.0","0.0","12.999999999999982","6.000000000000012","1.0658141036401503e-14","0.0"]}'
```

Expected prediction for digit is `4`.

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