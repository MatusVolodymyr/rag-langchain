# Use the official Miniconda3 image as the base
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file into the container
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Ensure the newly created environment is used:
ENV PATH /opt/conda/envs/rag_langchain/bin:$PATH

# Copy the rest of the project files into the container
COPY . .

# Expose the port that your FastAPI app will run on
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
