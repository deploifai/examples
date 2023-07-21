# Use an existing image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /SafeTweet

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY test.py .
COPY vectorizer.pkl .
COPY model.pkl .

# Set the default command to run when the container starts
CMD [ "python", "test.py" ]