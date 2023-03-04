FROM python:3.8
ADD main.py .
ADD requirements.txt .
# Generate requirements.txt
# python -m pipreqs.pipreqs
RUN pip install -r requirements.txt
CMD ["python", "./main.py"]