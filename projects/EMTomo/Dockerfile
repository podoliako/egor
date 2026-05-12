FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask numpy

COPY server.py viewer.html ./

CMD ["python", "server.py", "--runs-dir", "/runs", "--host", "0.0.0.0", "--port", "5050"]