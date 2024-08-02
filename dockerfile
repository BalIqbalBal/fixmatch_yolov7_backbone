# Gunakan base image yang sesuai, misalnya TensorFlow
FROM python:3.8-slim

# Instal dependensi tambahan jika diperlukan
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set working directory
WORKDIR /app

# Copy requirements.txt (jika ada)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh proyek ke dalam container
COPY . .

# Tentukan command untuk menjalankan aplikasi
CMD ["python", "main.py", "--pbar"]