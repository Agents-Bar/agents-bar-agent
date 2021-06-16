FROM python:3.8-slim
# set path to our python api file
ENV MODULE_NAME="app.main"

# install dependencies; for pytorch, install 1.9.0 as CPU
RUN pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install jsons
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./ /app

LABEL ai-traineree=v0.1.2
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
