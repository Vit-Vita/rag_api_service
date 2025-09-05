FROM python:3.9-slim

WORKDIR /code


COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run your FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
