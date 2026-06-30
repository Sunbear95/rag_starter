import { useState } from 'react'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')

  async function send(e) {
    e.preventDefault()
    if (!input.trim()) return

    setMessages((m) => [...m, { role: 'user', text: input }])
    setInput('')

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input }),
    })
    const data = await res.json()

    setMessages((m) => [...m, {
      role: 'assistant',
      text: data.reply,
      citations: data.citations || [],
      usage: data.usage || null,
      latencyMs: data.latency_ms ?? null,
    }])
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
        />
        <button type="submit">Send</button>
      </form>
    </div>
  )
}
