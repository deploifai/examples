# Use an existing image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /examples/diabetes_classifier

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY app.py .
COPY KNN_Classifier.pkl .


# Set the default command to run when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]