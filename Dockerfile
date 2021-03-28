
#######################################################
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 AS dev

# set path to our python api file
ENV MODULE_NAME="app.main"

# Install Poetry
RUN apt-get update && apt-get install curl -y
RUN apt-get -y install git

# install dependencies
COPY requirements.txt .

# Currently disabled as it may override laters pip install
# RUN pip install ...

# copy contents of project into docker
COPY ./ /app

#Expose port 80
EXPOSE 80

############################
FROM python:3.8-slim AS prod
# set path to our python api file
ENV MODULE_NAME="app.main"
COPY requirements.txt .

# install dependencies; for pytorch, install 1.8.1 as CPU
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.8.1%2Bcpu-cp38-cp38-linux_x86_64.whl
RUN pip install -r requirements.txt

COPY ./ /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
