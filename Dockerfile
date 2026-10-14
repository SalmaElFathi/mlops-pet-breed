FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY --chown=appuser:appuser params.yaml ./params.yaml
COPY --chown=appuser:appuser data/species_map.json ./data/species_map.json
COPY --chown=appuser:appuser models/model.pt ./models/model.pt
ENV PRODML_PARAMS_PATH=/app/params.yaml
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
