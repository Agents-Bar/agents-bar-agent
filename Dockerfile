FROM python:3.8-slim
# set path to our python api file
ENV MODULE_NAME="app.main"

# install dependencies; for pytorch, install 1.8.1 as CPU
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.8.1%2Bcpu-cp38-cp38-linux_x86_64.whl
COPY ./dist /dist
RUN pip install jsons
RUN pip install --no-index --find-links=/dist ai-traineree
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./ /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
