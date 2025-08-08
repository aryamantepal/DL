# -------- Step 1: pick a base image --------
# If training with GPU, pick CUDA base:
# ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
# If CPU only, change to:
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cpu

FROM ${BASE_IMAGE}

# -------- Step 2: set basic environment settings --------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# -------- Step 3: set working directory --------
WORKDIR /app

# -------- Step 4: install Python dependencies --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- Step 5: copy your code into the container --------
COPY . .

# -------- Step 6: set default environment variables --------
ENV DATA_DIR=/data \
    OUT_DIR=/outputs

# Make sure these folders exist inside the container
RUN mkdir -p /data /outputs

# -------- Step 7: set the default run command --------
CMD ["python", "main.py"]
