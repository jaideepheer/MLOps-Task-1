# MLOps-Task-1

Assignment (Will be graded)

1. Serve SVM and Decision tree models using the flask on separate relative URLs. i.e. `localhost:8000/svm_predict` and `localhost:8000/decision_tree_predict`. (ip could be different than localhost, in your case)

2. Dockerize the deployment i.e. create dockerfile and build image such that when you do `docker run` (may be with some more flags), the above two links should be accessible via curl. Write `docker_example.sh` shell script that includes the full curl commands.

## Run

### Full Sample

```
source docker_example.sh
```

### Start docker

```
source run_docker.sh
```

### CURL Command

```
curl -X 'POST' \
  'http://127.0.0.1:5000/decision_tree_predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image": [
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0
  ]
}'
```

## Output

### Full Sample

```
(/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1/.venv) jaideep@JD-GPC:/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1$ source docker_example.sh 
[+] Building 3.0s (11/11) FINISHED                                                                                                         
 => [internal] load build definition from Dockerfile                                                                                  0.0s
 => => transferring dockerfile: 38B                                                                                                   0.0s
 => [internal] load .dockerignore                                                                                                     0.0s
 => => transferring context: 2B                                                                                                       0.0s
 => [internal] load metadata for docker.io/library/python:3.8                                                                         2.7s
 => [internal] load build context                                                                                                     0.1s
 => => transferring context: 977B                                                                                                     0.1s
 => [1/6] FROM docker.io/library/python:3.8@sha256:68bddbf6e88c9c88d3238e13f02edf1884fc349a0964fad4b3d44f2425791ac7                   0.0s
 => CACHED [2/6] COPY requirements.txt /exp/requirements.txt                                                                          0.0s
 => CACHED [3/6] RUN pip3 install --no-cache-dir -r /exp/requirements.txt                                                             0.0s
 => CACHED [4/6] COPY mlops_task_1 /exp/mlops_task_1                                                                                  0.0s
 => CACHED [5/6] COPY api exp/api                                                                                                     0.0s
 => CACHED [6/6] WORKDIR /exp/api                                                                                                     0.0s
 => exporting to image                                                                                                                0.0s
 => => exporting layers                                                                                                               0.0s
 => => writing image sha256:a11471063e124a4e1dc5ae79421c6a61bb437bba1149d7829d372d4da25aa210                                          0.0s
 => => naming to docker.io/library/localtest:latest                                                                                   0.0s

Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
7d47b384170eedb371c307fbd57d39baa1455a4f62f0b977a73898287d7c493f
Sending request to decision_tree_predict...
{"prediction": 5}
Done.
Sending request to svm_predict...
{"prediction": 6}
Done.
localtestcnt
```

### SVM
```
(base) jaideep@JD-GPC:/mnt/c/Users/jaide$ curl -X 'POST' \
>   'http://127.0.0.1:5000/svm_predict' \
>   -H 'accept: application/json' \
>   -H 'Content-Type: application/json' \
>   -d '{
>   "image": [
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0
>   ]
> }'
{"prediction": 6}
```

### DTC
```
(base) jaideep@JD-GPC:/mnt/c/Users/jaide$ curl -X 'POST' \
>   'http://127.0.0.1:5000/decision_tree_predict' \
>   -H 'accept: application/json' \
>   -H 'Content-Type: application/json' \
 '{
  ">   -d '{
>   "image": [
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
,0.0,0.0>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0,
>     0.0,0.0,0.0,0.0
>   ]
> }'
{"prediction": 5}
```