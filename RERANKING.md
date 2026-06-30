# 리랭킹(Reranking) 작업 정리

병렬 작업/핸드오프용 문서. `feat/reranking` 브랜치 기준. 2026-06-30.

## 한 줄 요약

검색을 **2단계(dense 넓게 recall → cross-encoder로 재정렬 → 상위 소수만 LLM 투입)** 로 바꿔, 더 정밀하고 적은 청크로 답하게 함. **신규 의존성 없음**, 검색부만 수정해 인덱싱/청킹 작업과 충돌 없음.

---

## 왜 하나

기존은 dense bi-encoder(임베딩 코사인) top-5를 그대로 LLM에 넣었음. bi-encoder는 빠르지만 **정밀도가 낮아** 느슨하게 관련된 청크가 정답 청크보다 위로 올라오기도 함. cross-encoder는 `(query, chunk)`를 **함께 읽고** 관련도를 매기므로 훨씬 정확함. 다만 느려서 전체에 못 돌림 → **bi-encoder로 후보를 넓게 뽑고, 그 후보만 cross-encoder로 재정렬**하는 표준 2단계 구조.

## 무엇을 바꿨나 (파일 2개)

### 1. `indexer.py` — 리랭킹 함수 추가 (기존 `chunk_text`/`build_index` 등 청킹부는 **건드리지 않음**)

- 상수 `RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `get_reranker()` — `CrossEncoder` 지연 로딩(최초 1회 ~80MB 다운로드, 이후 캐시). `sentence-transformers`에 포함돼 **추가 설치 불필요**.
- `rerank(query, records, k)` — `(query, chunk_text)` 쌍을 cross-encoder로 점수화 → 내림차순 정렬 → 상위 k 반환.
- `search_rerank(query, records, k=4, candidates=20)` — 기존 `search()`로 `candidates`개를 dense recall한 뒤 `rerank()`로 상위 `k`만 추림.

### 2. `backend/app.py` — 검색 호출 교체

- `from indexer import load_index, search_rerank`
- `hits = search_rerank(user_message, INDEX, k=4, candidates=20)` (기존 `search(..., k=5)` 대체)

## 튜닝 손잡이

- `k` — LLM에 넣는 최종 청크 수 (기본 4). ↓면 토큰 절약, ↑면 완결성.
- `candidates` — cross-encoder가 보는 후보 풀 크기 (기본 20). ↑면 recall 여유(아래 한계 참고), cross-encoder 부담은 미미.
- `RERANK_MODEL` — 더 큰 `ms-marco-MiniLM-L-12-v2`로 바꾸면 정밀도↑/속도↓.

---

## 검증 결과 (현 FAA 인덱스, 4777청크 / dense k5 vs rerank k4)

| 질의 | 결과 |
|---|---|
| Flight review (§61.56) | ✅ **개선** — 인접 §청크(#144/145/146)를 묶어 더 완결된 컨텍스트 (dense는 #144,408,192,218로 흩어짐) |
| 산악지역 special VFR | ✅ **개선** — dense가 끌어온 `vol1` 잡음 4개를 제거하고 주제 적합한 Part 91/61 청크로 교체 |
| VFR 기상최저치 (Part 91) | ⚖️ 중립 — 둘 다 Part 91, 순서만 재정렬 |
| 의료증명 (private pilot) | ⚖️ 동일 — 둘 다 Part 61 (아래 "한계" 참고) |

**정직한 결론**: 이 FAA 인덱스에서 리랭킹의 실효는 *극적*이 아니라 **견실한 부분 개선**. 가장 큰 값어치는 **교차문서 노이즈 제거(vol1)** + **섹션 컨텍스트 묶음** + 토큰 소폭 절감(5→4).

## 알아둘 한계 — recall 문제는 리랭킹으로 못 고침

리랭킹은 *후보 풀을 재정렬*만 한다. **후보에 없는 청크는 절대 못 올린다.**

- "의료증명" 질의의 dense 후보 분포: top-20 = Part 61 ×20, **Part 67 = 0개** (top-40에서야 1개, top-80에서 2개).
- 즉 Part 67은 후보에 거의 안 들어와서 리랭킹이 손쓸 수 없음. (덧붙여 §61.23(Part 61)이 실제로 private pilot 의료증명 *요건*을 규정하므로 Part 61이 틀린 답도 아님.)
- **보완책**: `candidates`를 30~40으로 올려 recall 여유를 주거나, 임베딩/청킹 단계(동료 영역)에서 Part 67 청크의 검색성을 높이는 방향.

---

## 실행/테스트

```bash
# 백엔드 (이 브랜치 기준)
cd ~/ksept-lab/cowork/rag_starter/backend
HF_HUB_OFFLINE=1 ~/ksept-lab/rag-starter/.venv/bin/python app.py
```

dense vs rerank 직접 비교(파이썬):
```python
from indexer import load_index, search, search_rerank
idx = load_index()
q = "How often is a flight review required?"
print([f"{h['source']}#{h['chunk_index']}" for h in search(q, idx, k=5)])
print([f"{h['source']}#{h['chunk_index']}" for h in search_rerank(q, idx, k=4, candidates=20)])
```

## 남은 일 / 검토거리

- `candidates` 튜닝(20→30~40) 후 recall 개선 여부 재측정.
- **비교형 질의**(예: "Part 61 vs Part 91 차이")는 단일 쿼리 검색의 한계 — 리랭킹으로도 부분적. **쿼리 분해(multi-query)** 가 다음 레버.
- 정량 평가셋(FAA 질의 5~10개)으로 dense vs rerank 정확도/토큰 자동 비교.
