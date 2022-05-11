# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./main.py /code/
COPY ./models /code/models
COPY ./templates /code/templates

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
