FROM ubuntu:18.04
# copy src
COPY mlops_task_1 /exp/mlops_task_1
COPY requirements.txt /exp/requirements.txt
# install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
# set entrypoint
# my code creates the models folder
# RUN mkdir /exp/models
WORKDIR /exp
CMD ["python3", "./mlops_task_1/plot_graphs.py"]