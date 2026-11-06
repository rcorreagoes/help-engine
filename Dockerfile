FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema necessárias se aplicável
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código fonte
COPY . .

EXPOSE 8002

# Executar a aplicação (ajuste o caminho se necessário, ex: src.api:app)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8002"]