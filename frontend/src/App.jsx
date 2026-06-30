import { useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
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

  async function send(e) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setLoading(true)
    startTimer()

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question }),
      })
      const data = await res.json()
      setMessages((m) => [...m, {
        role: 'assistant',
        text: data.reply,
        citations: data.citations || [],
        usage: data.usage || null,
        latencyMs: data.latency_ms ?? null,
      }])
      setLastStats({ usage: data.usage || null, latencyMs: data.latency_ms ?? null })
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', text: `Error: ${err.message}` }])
    } finally {
      stopTimer()
      setLoading(false)
    }
  }

  // Session totals across all answered messages.
  const answered = messages.filter((m) => m.role === 'assistant' && m.usage)
  const sessionTokens = answered.reduce((sum, m) => sum + m.usage.total_tokens, 0)

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
      <div className="app">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg msg-${m.role}`}>
              <div className="msg-body">
                <b className="msg-role">{m.role}</b>
                {m.role === 'assistant' ? (
                  <div className="md">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                  </div>
                ) : (
                  <span className="msg-text">{m.text}</span>
                )}
              </div>
              {m.citations && m.citations.length > 0 && (
                <div className="sources">
                  Sources: {m.citations.map((c) => (
                    <span key={c.n} className="source">
                      [{c.n}] {c.source}
                    </span>
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
