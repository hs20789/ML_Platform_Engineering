#!/bin/sh
# 도커 컨테이너를 실행하는 명령어로 동작할 스크립트
uvicorn app.main:app --host 0.0.0.0 --port 8083