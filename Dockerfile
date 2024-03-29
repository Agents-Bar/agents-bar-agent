########################################################
# Base image
FROM python:3.8-slim as base

RUN pip install --upgrade pip~=22.0.3
# install dependencies; for pytorch, install 1.9.0 as CPU
RUN pip install --user torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Requirements in external file so that GitHub Dependabot shows vulnerabilities
COPY pyproject.toml pyproject.toml
COPY setup.cfg setup.cfg
RUN pip install --user -e .

########################################################
# Proper image
FROM python:3.8-slim

COPY --from=base /root/.local /root/.local
COPY ./ /app

# set path to our python api file
ENV PATH=/root/.local:/root/.local/bin:$PATH
ENV MODULE_NAME="app.main"

LABEL ai-traineree=v0.5.2

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
