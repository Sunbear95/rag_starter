# 작업 정리 (RAG starter)

이 문서는 병렬 작업/핸드오프용 변경 요약입니다. 2026-06-30 기준.

## 한 줄 요약

코퍼스가 **항공규정 14 CFR PDF 6종**으로 교체됨에 따라, 깨져 있던 수집 파이프라인을
**PDF 추출 + § 섹션 인식 청킹**으로 새로 구현. 청킹/파싱은 vol1(969p) 포함 전 문서에서
검증 완료(총 4,777 청크). **임베딩 빌드만 egress 정책으로 보류 중** (아래 블로커 참고).

---

## 배경: 왜 고쳤나

- 직전 커밋(`update documents`)에서 `documents/`가 Apollo 위키 `.md` → CFR `.pdf` 6종으로 교체됨.
- 그런데 `indexer.build_index()`는 `.md`/`.txt`만 처리 → **PDF는 전부 스킵 → 청크 0개**.
  즉 어떤 질문에도 "sources don't contain an answer"만 나오는 상태였음.
- 기존 청킹은 위키 짧은 제목 줄(`_looks_like_heading`)에 맞춰져 CFR의 `§ 61.1`,
  `Subpart A` 구조와 안 맞음.

## 직접 수정한 소스 파일

### 1. `indexer.py` — CFR PDF 수집 파이프라인 신설
- **컬럼 분할 추출**: CFR은 2단 레이아웃이라 단순 추출 시 좌/우 컬럼이 줄 단위로 섞임.
  페이지를 좌/우 절반으로 crop해 각 컬럼을 위→아래로 읽어 순서 복원 (pdfplumber).
- **헤더/푸터 제거**: 상·하단 밴드 crop + 러닝 헤더(`Federal Aviation Administration…`,
  `… CFR … Edition`, `VerDate…`) 정규식 제거.
- **텍스트 클린업**: 줄바꿈 하이픈 결합(`certifi-\ncate` → `certificate`), 줄바꿈→공백,
  Federal-Register 개정이력(`[… FR …]`)·`EFFECTIVE DATE NOTE` 노이즈 제거.
- **§ 섹션 인식 청킹**: `PART N—`, `Subpart X—`, `§ N.M Title.` 구조를 파싱.
  각 청크 앞에 `14 CFR Part 61 > Subpart A—General > § 61.1 Applicability and definitions`
  breadcrumb를 붙여 규정 구조를 임베딩·컨텍스트에 함께 실음.
- **TOC 중복 제거**: 각 Part 앞 목차가 섹션 제목을 중복 나열 → `(part, section)`별 본문이
  가장 긴 항목만 유지.
- **메타데이터 추가**: 레코드에 `part`, `subpart`, `section`(§ 번호), `section_title`,
  `page`(인쇄 페이지번호) 부착 → 추후 "차원2: 인용·근거"에서 `§ 91.113` 단위 인용에 사용.
- **긴 섹션 분할**: target ~1200자, overlap ~150자, 문장 경계 스냅.
- `.md`/`.txt` 범용 청커는 그대로 유지(자기 코퍼스 교체용).

### 2. `backend/requirements.txt`
- `pdfplumber>=0.11` 추가.

---

## 검증 결과 (임베딩 스텁, 파싱/청킹 로직)

빌드를 임베딩만 더미 벡터로 대체해 추출·파싱·청킹·스키마를 전수 검증:

| 문서 | 청크 |
|---|---|
| vol1 (969p, Part 1–59) | 3,509 |
| vol2-part61 | 520 |
| vol2-part67 | 64 |
| vol2-part71 | 21 |
| vol2-part73 | 14 |
| vol2-part91 | 649 |
| **합계** | **4,777** |

- 4,777/4,777 청크 전부 `section` + `page` 메타데이터 보유.
- vol1 샘플: `§ 26.21 | part 26 | page 483 | Limit of validity` — Part/섹션/페이지 정확.
- 추출 소요 ~187초 (오프라인 1회성).

---

## ⛔ 현재 블로커: 임베딩 모델 다운로드

- `index.pkl`를 실제로 빌드하려면 임베딩 모델
  (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)을
  HuggingFace에서 받아야 함.
- 이 세션의 egress 정책이 **`huggingface.co`를 차단(403)** → 모델 다운로드 불가.
  (정책 거부는 우회 금지 → 보고 대상.)
- **해소 방법**: 환경 네트워크 정책에서 `huggingface.co`(및 `cdn-lfs*.huggingface.co`)를
  허용하거나, 모델 캐시를 환경에 미리 심어두면 `python indexer.py`로 즉시 빌드 가능.
- 모델만 준비되면 코드 변경 없이 빌드가 통과하도록 검증해 둠.

## 실행 방법

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python indexer.py            # huggingface.co 접근 필요 (위 블로커)

# 백엔드 / 프론트엔드
cd backend && python app.py
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## 남은 일 / 다음 개선 차원

수집(차원0)은 코드 완료. 모델 접근만 풀리면 빌드·검색 검증 후 아래로 진행:

- **answer quality**: 하이브리드 검색(BM25+벡터), 리랭커, 멀티엔티티 질의 분해, 모델 ID 정정
  (`app.py`의 `claude-sonnet-4-6`는 실제 ID 아님).
- **citations & grounding**: `§` 번호+페이지 기반 인용(메타데이터 이미 준비됨), 근거 검증.
- **cost management**: 프롬프트 캐싱, 모델 티어링, 동적 k.
- **clarity**: 스트리밍, 답변 구조화.
- **robustness & safety**: `/api/chat` 에러 처리·입력 검증, 빈 결과 가드, 규제 도메인 디스클레이머.
- **user experience**: 멀티턴 히스토리(현재 단발성), 출처 미리보기.
