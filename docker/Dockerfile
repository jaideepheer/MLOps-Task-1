FROM python:3.8
# Install req
COPY requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
# copy src
COPY mlops_task_1 /exp/mlops_task_1
COPY api exp/api
EXPOSE 5000
# set entrypoints
WORKDIR /exp/api
CMD ["flask", "run", "--host=0.0.0.0"]