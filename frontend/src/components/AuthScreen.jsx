import { useState } from 'react'
import { ArrowRight, KeyRound, LoaderCircle, Mail, ShieldCheck } from 'lucide-react'

export default function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const response = await fetch(`/api/auth/${mode === 'login' ? 'login' : 'signup'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || 'Authentication failed')
      onAuthenticated(body)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-page">
      <div className="auth-grid-glow" />
      <section className="auth-pitch">
        <div className="brand-lockup auth-brand"><div className="brand-mark">R</div><div><strong>RepoMind</strong><span>codebase intelligence</span></div></div>
        <div className="auth-kicker"><span className="status-pip" /> Private by design</div>
        <h1>Your codebase,<br /><em>understood.</em></h1>
        <p>A focused workspace for exploring architecture, tracing behavior, and asking better questions of your repositories.</p>
        <div className="auth-proof"><ShieldCheck size={16} /><span>Every account has its own conversations and repository workspace.</span></div>
      </section>
      <section className="auth-card">
        <div className="auth-card-heading"><div className="auth-icon"><KeyRound size={18} /></div><div><div className="eyebrow">Workspace access</div><h2>{mode === 'login' ? 'Welcome back' : 'Create your workspace'}</h2></div></div>
        <div className="auth-tabs"><button className={mode === 'login' ? 'auth-tab-active' : ''} onClick={() => { setMode('login'); setError('') }}>Log in</button><button className={mode === 'signup' ? 'auth-tab-active' : ''} onClick={() => { setMode('signup'); setError('') }}>Sign up</button></div>
        <form className="auth-form" onSubmit={submit}>
          <label>Email address<div className="auth-input"><Mail size={16} /><input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" autoComplete="email" required /></div></label>
          <label>Password<div className="auth-input"><KeyRound size={16} /><input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="At least 8 characters" minLength="8" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required /></div></label>
          {error && <div className="auth-error">{error}</div>}
          <button className="auth-submit" type="submit" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <ArrowRight size={16} />}{mode === 'login' ? 'Enter workspace' : 'Create account'}</button>
        </form>
        <p className="auth-legal">Your account keeps your conversations separate from every other workspace.</p>
      </section>
    </main>
  )
}