FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.fileWatcherType", "none"] 