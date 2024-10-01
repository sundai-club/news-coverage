# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENV STREAMLIT_SERVER_PORT=8080

# Run streamlit when the container launches
CMD streamlit run app.py  --server.enableCORS false --server.enableXsrfProtection false --server.headless=true --server.maxUploadSize 1000
