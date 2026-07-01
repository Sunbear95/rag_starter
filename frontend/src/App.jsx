import { useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [concise, setConcise] = useState(false) // ask the backend for shorter, cheaper answers
  const [elapsedMs, setElapsedMs] = useState(0)
  const [lastStats, setLastStats] = useState(null) // { usage, latencyMs } of the latest answer
  const timerRef = useRef(null)

  function startTimer() {
    const start = Date.now()
    setElapsedMs(0)
    timerRef.current = setInterval(() => setElapsedMs(Date.now() - start), 100)
  }

  function stopTimer() {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  // Applies an update fn to just the last message in state (the in-flight
  // assistant placeholder). Safe because the form stays disabled — no other
  // message can be appended — between the placeholder push and stream end.
  function patchLastMessage(update) {
    setMessages((m) => {
      const next = [...m]
      next[next.length - 1] = { ...next[next.length - 1], ...update(next[next.length - 1]) }
      return next
    })
  }

  async function send(e) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setLoading(true)
    startTimer()
    setMessages((m) => [...m, { role: 'assistant', text: '', citations: [], retrieved: [], toolCalls: [], usage: null, latencyMs: null }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, concise }),
      })

      if (!res.ok) {
        // Pre-stream failure (validation error, rate limit, ...) — plain JSON body.
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || `Request failed (${res.status})`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let gotFirstEvent = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        // { stream: true } holds back a multi-byte UTF-8 sequence split across
        // chunk boundaries instead of emitting replacement characters.
        buffer += decoder.decode(value, { stream: true })

        let newlineIndex
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, newlineIndex)
          buffer = buffer.slice(newlineIndex + 1)
          if (!line.trim()) continue
          const event = JSON.parse(line)

          // Either a delta or a tool_call marks the end of "waiting" — the
          // model is visibly doing something now, whichever comes first.
          if ((event.type === 'delta' || event.type === 'tool_call') && !gotFirstEvent) {
            gotFirstEvent = true
            stopTimer()
          }

          if (event.type === 'delta') {
            patchLastMessage((last) => ({ text: last.text + event.text }))
          } else if (event.type === 'tool_call') {
            // Any text streamed so far this turn was pre-tool-call preamble
            // (e.g. "Let me search for that!") the model emitted before
            // deciding to call the tool — the backend only sends tool_call
            // after that iteration's text has fully streamed, so it's safe
            // to drop here. It never belongs in the final answer.
            patchLastMessage((last) => ({
              text: '',
              toolCalls: [...(last.toolCalls || []), { name: event.name, query: event.query, resultCount: event.result_count }],
            }))
          } else if (event.type === 'done') {
            patchLastMessage(() => ({
              citations: event.citations || [],
              retrieved: event.retrieved || [],
              usage: event.usage || null,
              latencyMs: event.latency_ms ?? null,
            }))
            setLastStats({ usage: event.usage || null, latencyMs: event.latency_ms ?? null })
          } else if (event.type === 'error') {
            patchLastMessage((last) => ({ text: `${last.text}\n\n_Error: ${event.message}_` }))
          }
        }
      }
    } catch (err) {
      patchLastMessage((last) => ({ text: last.text || `Error: ${err.message}` }))
    } finally {
      stopTimer()
      setLoading(false)
    }
  }

  // Session totals across all answered messages.
  const answered = messages.filter((m) => m.role === 'assistant' && m.usage)
  const sessionTokens = answered.reduce((sum, m) => sum + m.usage.total_tokens, 0)

  // Retrieval detail (search keywords + chunks) for the latest answer, shown in
  // the left panel instead of inline in the chat so the conversation stays clean.
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')
  const retrievalCited = new Set((lastAssistant?.citations || []).map((c) => c.n))

  // Tools invoked for the latest answer, collapsed to per-tool call counts so
  // the right panel shows *which* tools ran (and how often), not every query.
  const toolUsage = Object.entries(
    (lastAssistant?.toolCalls || []).reduce((acc, t) => {
      const name = t.name || 'tool'
      acc[name] = (acc[name] || 0) + 1
      return acc
    }, {})
  )

  return (
    <div className="page">
      <header className="faa-header">
        <div className="faa-brand">
          <span className="faa-mark" aria-hidden="true">✈</span>
          <span className="faa-word">FAA</span>
          <span className="faa-sub">Docs Assistant</span>
        </div>
        <div className={`faa-status ${loading ? 'on' : ''}`}>
          <span className="dot" aria-hidden="true" />
          {loading ? 'working' : 'live'}
        </div>
      </header>

      <div className="layout">
      <aside className="retrieval-panel" aria-live="polite">
        <h2>Retrieval</h2>
        <div className="retrieval-body">
          {lastAssistant?.toolCalls && lastAssistant.toolCalls.length > 0 && (
            <div className="retrieval-group">
              <div className="retrieval-title">Search keywords</div>
              <div className="tool-calls">
                {lastAssistant.toolCalls.map((t, ti) => (
                  <div key={ti} className="tool-call">
                    🔍 <em>{t.query}</em>
                    {t.resultCount > 0 ? ` (${t.resultCount} new result${t.resultCount === 1 ? '' : 's'})` : ' (no new results)'}
                  </div>
                ))}
              </div>
            </div>
          )}

          {lastAssistant?.retrieved && lastAssistant.retrieved.length > 0 ? (
            <div className="retrieval-group">
              <div className="retrieval-title">Retrieved chunks ({lastAssistant.retrieved.length})</div>
              {lastAssistant.retrieved.map((r) => (
                <details key={r.n} className={`rchunk ${retrievalCited.has(r.n) ? 'rchunk-cited' : ''}`}>
                  <summary className="rchunk-head">
                    <span className="rchunk-n">[{r.n}]</span> {r.source} · chunk #{r.chunk_index}
                    {retrievalCited.has(r.n) && <span className="rchunk-badge">cited</span>}
                  </summary>
                  <div className="rchunk-text">{r.text}</div>
                </details>
              ))}
            </div>
          ) : (
            (!lastAssistant?.toolCalls || lastAssistant.toolCalls.length === 0) && (
              <div className="retrieval-empty">Ask a question to see the search keywords and retrieved chunks here.</div>
            )
          )}
        </div>
      </aside>

      <div className="app">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg msg-${m.role}`}>
              <div className="msg-body">
                <b className="msg-role">{m.role}</b>
                {m.role === 'assistant' ? (
                  <div className="md">
                    {m.text ? (
                      // Partial markdown mid-stream can render roughly (e.g. an
                      // unclosed code fence or table row) — resolves once the
                      // stream completes and the full text re-parses cleanly.
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                    ) : (
                      i === messages.length - 1 && loading && <span className="thinking">…</span>
                    )}
                  </div>
                ) : (
                  <span className="msg-text">{m.text}</span>
                )}
              </div>
              {m.citations && m.citations.length > 0 && (
                <div className="sources">
                  <span className="sources-label">Sources</span>
                  {m.citations.map((c) => (
                    <details key={c.n} className="source">
                      <summary>
                        [{c.n}] {c.source}
                        {c.section ? ` · ${c.section}` : ''}
                        {c.page != null ? ` · p.${c.page}` : ''}
                      </summary>
                      <div className="source-excerpt">
                        {c.section_title && <div className="source-title">{c.section_title}</div>}
                        {c.excerpt}
                      </div>
                    </details>
                  ))}
                </div>
              )}
              {(m.usage || m.latencyMs != null) && (
                <div className="usage">
                  {m.usage && `Tokens: ${m.usage.total_tokens} (in ${m.usage.input_tokens} / out ${m.usage.output_tokens})`}
                  {m.usage && m.latencyMs != null && ' · '}
                  {m.latencyMs != null && `${(m.latencyMs / 1000).toFixed(2)}s`}
                </div>
              )}
            </div>
          ))}
        </div>
        <form onSubmit={send}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the indexed docs..."
            autoFocus
            disabled={loading}
          />
          <label className="concise-toggle" title="Short answers in Rocky's voice — fewer tokens">
            <input
              type="checkbox"
              checked={concise}
              onChange={(e) => setConcise(e.target.checked)}
              disabled={loading}
            />
            Rocky 🪨
          </label>
          <button type="submit" disabled={loading}>{loading ? '…' : 'Send'}</button>
        </form>
      </div>

      <aside className="stats-panel" aria-live="polite">
        <h2>Stats</h2>
        <div className="stats-body">
          <div className="stat-status">
            {loading
              ? <span className="stat-live">⏱ {(elapsedMs / 1000).toFixed(1)}s</span>
              : <span className="stat-idle">idle</span>}
          </div>

          <div className="stat-group">
            <div className="stat-title">Last answer</div>
            <div className="stat-row">
              <span className="stat-key">Time</span>
              <span className="stat-val">{lastStats?.latencyMs != null ? `${(lastStats.latencyMs / 1000).toFixed(2)}s` : '—'}</span>
            </div>
            <div className="stat-row">
              <span className="stat-key">Tokens</span>
              <span className="stat-val">{lastStats?.usage ? lastStats.usage.total_tokens : '—'}</span>
            </div>
            <div className="stat-row stat-dim">
              <span className="stat-key">in / out</span>
              <span className="stat-val">{lastStats?.usage ? `${lastStats.usage.input_tokens} / ${lastStats.usage.output_tokens}` : '—'}</span>
            </div>
            {lastStats?.usage?.cache_read_tokens > 0 && (
              <div className="stat-row stat-dim">
                <span className="stat-key">cache read</span>
                <span className="stat-val">{lastStats.usage.cache_read_tokens}</span>
              </div>
            )}
          </div>

          <div className="stat-group">
            <div className="stat-title">Tools used</div>
            {toolUsage.length > 0 ? (
              toolUsage.map(([name, count]) => (
                <div key={name} className="stat-row">
                  <span className="stat-key">🔧 {name}</span>
                  <span className="stat-val">{count}×</span>
                </div>
              ))
            ) : (
              <div className="stat-row stat-dim">
                <span className="stat-key">none yet</span>
                <span className="stat-val">—</span>
              </div>
            )}
          </div>

          <div className="stat-group">
            <div className="stat-title">Session</div>
            <div className="stat-row">
              <span className="stat-key">Requests</span>
              <span className="stat-val">{answered.length}</span>
            </div>
            <div className="stat-row">
              <span className="stat-key">Total tokens</span>
              <span className="stat-val">{sessionTokens}</span>
            </div>
          </div>
        </div>
      </aside>
      </div>
    </div>
  )
}
