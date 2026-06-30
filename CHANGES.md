# 작업 정리 (RAG starter)

이 문서는 병렬 작업/핸드오프용 변경 요약입니다. 2026-06-30 기준.

## 한 줄 요약

청킹 품질 개선 + 인덱스 빌드 + 답변별 **토큰 사용량 및 소요 시간** 표시 추가. 코드·인덱스 준비 완료, 바로 실행 가능.

---

## 직접 수정한 소스 파일 (4개)

### 1. `indexer.py` — `chunk_text()` 개선
- **breadcrumb 프리픽스**: 각 청크 앞에 `문서제목 > 섹션제목`을 붙임 (예: `Apollo 11 > Lunar surface operations`). 본문에 미션명이 없어도 임베딩/컨텍스트에 맥락이 실려 미션명 기반 질의 매칭이 좋아짐.
- **섹션 인식 청킹**: 위키 추출본의 섹션 제목(짧은 단독 라인)을 감지해 섹션 경계에서 청크를 나눔.
- **문장 경계 overlap**: 기존 `current[-100:]` 단순 컷 → 문장/단어 경계에 스냅.
- 보조 함수 추가: `_looks_like_heading()`, `_overlap_tail()`, `_doc_title()`.
- `build_index()`가 문서 제목(첫 `# H1`)을 뽑아 `chunk_text(doc_title=...)`로 전달.

### 2. `backend/app.py` — 토큰 사용량 + 소요 시간 응답 추가
- `/api/chat` 응답에 `usage` 필드 추가: `{input_tokens, output_tokens, total_tokens}` (`resp.usage`에서 추출).
- `/api/chat` 응답에 `latency_ms` 필드 추가: `time.perf_counter()`로 요청 시작~응답까지(검색 + LLM 호출) end-to-end 측정. `import time` 추가.
- 검색 `k`는 현재 **5** (기본값). 토큰/재현율 트레이드오프 검토거리.

### 3. `frontend/src/App.jsx` — 토큰 + 시간 표시
- 응답에서 `usage`와 `latency_ms`를 받아 메시지 상태에 저장.
- 각 assistant 답변 아래 `Tokens: N (in X / out Y) · 1.23s` 형식으로 표시.

### 4. `frontend/src/index.css` — 표시 스타일
- `.usage` 클래스 추가 (작은 회색 모노스페이스, 기존 `.sources` 톤에 맞춤). 토큰·시간 한 줄에 같이 표시.

---

## 삭제한 파일 (1개)

- `documents/06-changelog.md` — Apollo가 아니라 습관 추적 앱 changelog. 스타터 템플릿 잔재(다른 세션 영향 아님). 검색 노이즈라 제거 → 현재 Apollo 문서 정확히 20개.

---

## 자동 생성/부산물 (신경 안 써도 됨)

- `index.pkl` — 인덱스 빌드 결과 (1100 청크, 4.6MB)
- `__pycache__/`, `backend/__pycache__/` — 파이썬 캐시
- `frontend/package-lock.json` — `npm install` 부산물
- `.claude/settings.local.json` — 에이전트 설정

---

## 현재 환경 상태

- `.venv` 생성됨 (python3.11), `backend/requirements.txt` 설치 완료.
- 임베딩 모델 다운로드 완료, `index.pkl` 빌드 완료 (1100 청크).
- API 키: 별도 `.env` 없이 `~/ksept-lab/.env`의 `ANTHROPIC_API_KEY`를 `load_dotenv()`가 상위 탐색으로 자동 로드함.

## 실행 방법

```bash
# 백엔드 (터미널 1)
cd ~/ksept-lab/rag-starter && source .venv/bin/activate && cd backend && python app.py

# 프론트엔드 (터미널 2)
cd ~/ksept-lab/rag-starter/frontend && npm install && npm run dev
# http://localhost:5173
```

> curl 테스트 시 `localhost` 대신 `127.0.0.1` 사용 (Flask가 IPv4에만 바인딩).

## 검증 결과 (참고)

- "Apollo 1 화재 원인" → 정확, 출처 `apollo-01.md`.
- "어느 미션이 달에 착륙했나" → usage in=1432/out=180, latency≈9.5s (k=5 기준).
- "11호 vs 17호 moonwalk 비교" → **약함**: 단일 벡터 검색에서 두 엔티티가 희석돼 11/17이 안 잡힘. (보고서의 "실패 사례"로 적합)

## 남은 일 / 검토거리

- 검색 `k` 값 튜닝 (토큰 비용 ↔ 재현율).
- README 4·"What to present" 항목용 보고서 작성: 테스트 질문 5개, 강점 2/약점 2, 청킹 선택 근거, 실패 사례 + 개선책.
