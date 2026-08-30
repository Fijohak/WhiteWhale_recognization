FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY requirements-platform.txt ./
RUN pip install --no-cache-dir -r requirements-platform.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src

RUN useradd --system --uid 10001 --create-home whitewhale \
    && mkdir -p /srv/whitewhale/data \
    && chown -R whitewhale:whitewhale /srv/whitewhale /app
USER whitewhale

EXPOSE 8000
CMD ["sh", "-c", "python -m alembic upgrade head && exec uvicorn whitewhale.platform.main:app --host 0.0.0.0 --port 8000"]
