FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch: far smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "bot.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
