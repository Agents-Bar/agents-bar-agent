FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# set path to our python api file
ENV MODULE_NAME="app.main"

# install poetry
RUN pip install poetry

# disable virtualenv for peotry
RUN poetry config virtualenvs.create false
COPY pyproject.toml .

# install dependencies
RUN poetry install

# copy contents of project into docker
COPY ./ /app

