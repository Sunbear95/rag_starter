import { useState, useRef } from 'react'

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

  // Live timer while generating; the latest answer's time + tokens when idle.
  let indicator = null
  if (loading) {
    indicator = `⏱ ${(elapsedMs / 1000).toFixed(1)}s`
  } else if (lastStats) {
    const t = lastStats.latencyMs != null ? `${(lastStats.latencyMs / 1000).toFixed(2)}s` : ''
    const u = lastStats.usage
      ? `${lastStats.usage.total_tokens} tok (in ${lastStats.usage.input_tokens} / out ${lastStats.usage.output_tokens})`
      : ''
    indicator = [t, u].filter(Boolean).join(' · ')
  }

  return (
    <div className="app">
      <h1>RAG Chat</h1>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            <div className="msg-body"><b>{m.role}:</b> {m.text}</div>
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
        <div className={`indicator ${loading ? 'indicator-live' : ''}`} aria-live="polite">
          {indicator}
        </div>
      </form>
    </div>
  )
}
