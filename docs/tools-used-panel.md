# "Tools used" 패널 (사용된 툴 표시)

동료 핸드오프용 문서. 채팅 오른쪽 **Stats 패널**에 이번 답변에서 **어떤 툴이 몇 번 호출됐는지**
보여주는 "Tools used" 섹션을 추가한 작업 정리입니다.

## 한 줄 요약

모델이 답변 중 호출한 툴(현재는 `search_cfr` 하나)의 **이름과 호출 횟수**를 오른쪽 패널에
집계해 표시. 예: `🔧 search_cfr   5×`. 왼쪽 "Retrieval" 패널이 *무엇을* 검색했는지(키워드·청크)를
보여준다면, 이 패널은 *어떤 툴이 몇 번* 돌았는지를 한눈에 보여줍니다.

## 동작

1. 모델이 한 턴에서 `search_cfr` 툴을 호출할 때마다 백엔드가 `tool_call` 스트림 이벤트를 방출.
2. 프론트가 이 이벤트를 마지막 assistant 메시지의 `toolCalls` 배열에 쌓음.
3. 툴 이름별로 호출 횟수를 집계해 오른쪽 Stats 패널의 "Tools used" 그룹에 렌더.
4. 아직 툴을 안 쓴 상태면 `none yet —` 표시.

> 참고: 한 턴의 검색 횟수는 `MAX_SEARCHES_PER_TURN` 상한이 걸려 있어(현재 5),
> 복잡한 질문이면 `search_cfr 5×`까지 올라갑니다.

## 변경한 파일

### `backend/app.py`
- `tool_call` 이벤트에 `name` 필드 추가 (`block.name`). 프론트가 툴 이름을 하드코딩하지 않고
  백엔드가 준 이름을 그대로 쓰도록 함 → 나중에 툴이 늘어나도 프론트 수정 불필요.

```python
yield json.dumps({
    "type": "tool_call",
    "name": block.name,          # ← 추가
    "query": query,
    "result_count": len(new_hits),
}) + "\n"
```

### `frontend/src/App.jsx`
- 스트림 처리에서 `toolCalls`에 `name`을 함께 저장.
- 마지막 답변의 tool_call을 툴 이름별 횟수로 집계하는 `toolUsage` 계산 추가.
- 오른쪽 Stats 패널에 "Tools used" `stat-group` 추가 (기존 `stat-*` CSS 재사용, 새 CSS 없음).

## 확인 방법

```bash
source .venv/bin/activate
python backend/app.py                 # 포트 5000 (macOS AirPlay가 5000을 점유하면 끄거나 포트 변경)
cd frontend && npm run dev            # http://localhost:5173
```

http://localhost:5173 에서 질문 → 오른쪽 패널 "Tools used"에 호출된 툴/횟수가 뜹니다.

백엔드만 빠르게 확인하려면:

```bash
curl -sN http://127.0.0.1:5000/api/chat -X POST \
  -H 'Content-Type: application/json' \
  -d '{"message":"What are the medical certificate requirements for a private pilot?"}' \
  | grep tool_call
# → "type": "tool_call", "name": "search_cfr", "query": "...", "result_count": 5} ...
```

## 툴을 새로 추가할 때

패널은 백엔드가 보내는 `name`을 그대로 그룹핑하므로, 새 툴을 `tools=[...]`에 추가하고
호출 시 동일하게 `tool_call` 이벤트(`name` 포함)를 방출하기만 하면 **프론트 수정 없이**
자동으로 "Tools used"에 집계됩니다.
