# Use an existing image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /SafeTweet

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY app.py .
COPY vectorizer.pkl .
COPY model.pkl .

# Set the default command to run when the container starts
CMD [ "python", "app.py" ]