FROM python:3.12-slim

# Usuário não-root (exigido pelo Hugging Face Spaces, boa prática em qualquer lugar)
RUN useradd -m -u 1000 app
WORKDIR /home/app/api

COPY --chown=app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app app ./app
COPY --chown=app artifacts ./artifacts

USER app
ENV PYTHONUNBUFFERED=1
# Render injeta $PORT; Hugging Face Spaces usa 7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
