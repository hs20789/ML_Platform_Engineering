FROM python:3.11-slim   

WORKDIR /joker

ENV PATH="/joker/.venv/bin:${PATH}"
ENV UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/  

COPY . .

RUN uv sync --locked --no-dev --no-install-project

EXPOSE 8083

RUN chmod +x /joker/entrypoint.sh
ENTRYPOINT ["/joker/entrypoint.sh"]