# 간결 모드 (로키 보이스) 변경 정리

이 문서는 `feature/concise-rocky-mode` 브랜치의 변경 요약입니다. 2026-07-01 기준.

## 한 줄 요약

출력(답변) 토큰을 줄이기 위한 **선택형 간결 모드**를 추가. 켜면 답변이 훨씬 짧아지고,
말투는 소설 『프로젝트 헤일메리』의 외계인 **로키(Rocky)** 스타일(단순·투박·친근)로 나옴.
**근거·인용 규칙은 그대로 유지**.

---

## 배경: 왜 만들었나

- 기존 답변이 길어 출력 토큰을 많이 소비 → 짧고 저렴하게 답하는 모드가 필요했음.
- 단순히 짧게만 하지 않고, 사용자 요청으로 답변 말투를 로키 보이스로 스타일링.

## 직접 수정한 소스 파일

### 1. `backend/app.py` — 간결 모드 백엔드

- **`concise` 플래그 수신**: `/api/chat` 요청 본문에서 `concise`(bool, 기본 `false`)를 읽음.
- **`CONCISE_INSTRUCTION` 추가**: 켜졌을 때만 시스템 프롬프트에 **두 번째(비캐시) 블록**으로
  덧붙임. 기존 베이스 프롬프트의 캐시 엔트리는 두 모드가 그대로 공유 → 캐시 효율 유지.
- **로키 보이스 규칙**: 짧은 문장·현재형·관사 생략, `"you"`로 말 걸기, 가볍게 깨진 문법,
  질문은 `"Question: ..."`으로 시작, 감정은 한 단어(`amaze`/`sad`/`scared`).
  **중간 단어 반복 금지**, 대신 답변 **마지막 줄에만 `"Amaze, amaze, amaze."`**.
  이 톤을 **모든 줄(나열·경고 포함)** 에 적용하도록 목표 예시를 프롬프트에 명시.
- **토큰 상한**: `max_tokens`를 간결 모드에서 `4096 → 1024`로 낮춤
  (`MAX_TOKENS_DEFAULT` / `MAX_TOKENS_CONCISE`). 상한은 안전 바운드이고, 실제로 답을
  짧게 만드는 것은 지시문.
- **근거·인용 유지**: 툴 결과 기반, `[n]` 인용 필수 규칙은 간결 모드에서도 그대로.
  사용자 언어에 맞춤(한국어면 한국어로 로키 스타일).

### 2. `frontend/src/App.jsx` — 토글 UI

- `concise` 상태 추가, `/api/chat` 요청 본문에 함께 전송.
- 입력창 옆에 **`Rocky 🪨` 체크박스** 추가(툴팁: 짧은 답변·토큰 절약).

### 3. `frontend/src/index.css` — 체크박스 스타일

- `.concise-toggle` 전용 스타일 추가. 제네릭 `input { flex: 1 }`에 체크박스가
  늘어나지 않도록 격리(`flex: 0 0 auto`, `width: auto`).

---

## 동작 예시

간결 모드 ON, "음주 후 비행 규정" 질문에 대한 예상 답변:

```
Pilot cannot fly if body still feel alcohol [2]. You drink today? Then you wait
8 hours before fly [2]. Blood or breath alcohol 0.04 or more? No fly [2].

Careful: law officer ask for alcohol test, you say no? You lose certificate [1].

Simple: you drink today, wait 8 hours, check body feel okay, then fly [2].

Amaze, amaze, amaze.
```

---

## 사용 방법

- UI: 입력창 옆 **`Rocky 🪨`** 체크박스를 켜고 질문.
- API 직접 호출:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "음주 후 비행 규정?", "concise": true}'
```

- `.py` 변경이라 반영하려면 **백엔드 재시작** 필요.

## 참고

- 이 모드는 **출력 토큰**을 줄임. 입력 토큰(검색된 CFR 청크)은 그대로라 인용 근거는 유지됨.
- 검색 지시(최소 1회 검색 등)는 그대로 두어 근거 기반 답변 품질을 유지.
