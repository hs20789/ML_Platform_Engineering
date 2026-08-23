# Chapter 3 — Hello Joker

FastAPI로 만든 간단한 "농담 반환" API를 Docker 이미지로 빌드하고, Kubernetes Pod로 배포해보는 실습 예제입니다.

## 구성

- `app/` — FastAPI 애플리케이션 (`GET /` 요청 시 [pyjokes](https://pypi.org/project/pyjokes/) 기반 랜덤 농담을 반환)
- `Dockerfile`, `entrypoint.sh` — [uv](https://docs.astral.sh/uv/)를 사용한 애플리케이션 컨테이너 이미지 빌드/실행 설정
- `k8s/` — 이미지를 배포하기 위한 Kubernetes 매니페스트
- `pyproject.toml`, `uv.lock` — 의존성 정의 및 잠금 파일

## 로컬 실행

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8083
```

## Docker

```bash
docker build -t heons/hello-joker:v1 .
docker run -p 8083:8083 heons/hello-joker:v1
```

## Kubernetes

```bash
kubectl apply -f k8s/pod.yaml
```
