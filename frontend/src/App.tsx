import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// Mirrors the backend's TriageResponse schema.
type TriageResponse = {
  result: {
    category: 'refund' | 'shipping' | 'warranty' | 'other'
    priority: 'low' | 'medium' | 'high'
    summary: string
    suggested_reply: string
    policy_found: boolean
  }
  sources: { document: string; section: string; score: number }[]
  mode: 'openai' | 'mock'
}

const EXAMPLES = [
  'My headphones arrived damaged and the box was crushed. Can I get my money back?',
  'I ordered 12 days ago and the parcel still has not arrived.',
  'My speaker battery stopped holding charge after 8 months. Is that covered?',
  'Do you sell gift cards?',
]

export default function App() {
  const [message, setMessage] = useState('')
  const [data, setData] = useState<TriageResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit() {
    setLoading(true)
    setError('')
    setData(null)
    try {
      const res = await fetch(`${API_URL}/triage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })
      const body = await res.json()
      if (!res.ok) {
        // FastAPI sends a string for our errors and a list for validation (422) errors.
        setError(typeof body.detail === 'string' ? body.detail : 'Please write at least 5 characters.')
        return
      }
      setData(body)
    } catch {
      setError('Could not reach the server. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main>
      <h1>Support message triage</h1>
      <p className="muted">Paste a customer message. The AI classifies it and drafts a reply using only the shop's policies.</p>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button key={example} className="chip" onClick={() => setMessage(example)}>
            {example.slice(0, 32)}…
          </button>
        ))}
      </div>

      <textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="e.g. My order arrived broken, what can I do?"
        rows={4}
      />
      <button className="primary" onClick={submit} disabled={loading || !message.trim()}>
        {loading ? 'Analysing…' : 'Analyse message'}
      </button>

      {error && <div className="error">{error}</div>}

      {data && (
        <section className="card">
          <div className="badges">
            <span className="badge">{data.result.category}</span>
            <span className={`badge priority-${data.result.priority}`}>{data.result.priority} priority</span>
            {!data.result.policy_found && <span className="badge warn">no matching policy</span>}
            <span className="badge mode">{data.mode === 'mock' ? 'mock mode' : 'OpenAI'}</span>
          </div>
          <h2>Summary</h2>
          <p>{data.result.summary}</p>
          <h2>Suggested reply</h2>
          <p className="reply">{data.result.suggested_reply}</p>
          <h2>Based on</h2>
          {data.sources.length === 0 ? (
            <p className="muted">No policy section matched.</p>
          ) : (
            <ul>
              {data.sources.map((s) => (
                <li key={s.section}>
                  {s.document} → {s.section} <span className="muted">(score {s.score})</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </main>
  )
}
