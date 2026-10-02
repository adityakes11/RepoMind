import { useEffect, useState } from 'react'
import { AlertCircle, LoaderCircle, LogOut, Menu, X } from 'lucide-react'
import { api } from './api/client'
import AuthScreen from './components/AuthScreen'
import ChatWindow from './components/ChatWindow'
import IngestForm from './components/IngestForm'
import Sidebar from './components/Sidebar'

export default function App() {
  const [user, setUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [threads, setThreads] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [thread, setThread] = useState(null)
  const [repoUrl, setRepoUrl] = useState('')
  const [ingestStatus, setIngestStatus] = useState(null)
  const [ingestError, setIngestError] = useState('')
  const [question, setQuestion] = useState('')
  const [streaming, setStreaming] = useState(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    if (!localStorage.getItem('repomind_token')) { setAuthLoading(false); return }
    api.me().then(setUser).catch(() => localStorage.removeItem('repomind_token')).finally(() => setAuthLoading(false))
  }, [])

  const refreshThreads = async (selectFirst = true) => {
    const nextThreads = await api.listThreads()
    setThreads(nextThreads)
    if (selectFirst && !activeId && nextThreads[0]) setActiveId(nextThreads[0].id)
    return nextThreads
  }

  useEffect(() => {
    if (!user) return
    refreshThreads().catch((err) => setError(err.message)).finally(() => setBusy(false))
  }, [user])

  useEffect(() => {
    if (!activeId) { setThread(null); return }
    setBusy(true)
    api.getThread(activeId).then((nextThread) => {
      setThread(nextThread)
      setRepoUrl(nextThread.github_url || '')
      setIngestStatus(null)
      setError('')
    }).catch((err) => setError(err.message)).finally(() => setBusy(false))
    setSidebarOpen(false)
  }, [activeId])

  const createNewThread = async () => {
    try {
      const nextThread = await api.createThread()
      setThreads((current) => [nextThread, ...current])
      setActiveId(nextThread.id)
      setRepoUrl('')
      setError('')
    } catch (err) { setError(err.message) }
  }

  const deleteCurrentThread = async (id) => {
    try {
      await api.deleteThread(id)
      const remaining = threads.filter((item) => item.id !== id)
      setThreads(remaining)
      if (id === activeId) setActiveId(remaining[0]?.id || null)
    } catch (err) { setError(err.message) }
  }

  const ingest = async (event) => {
    event.preventDefault()
    setIngestError('')
    try {
      let target = thread
      if (!target) {
        target = await api.createThread()
        setThreads((current) => [target, ...current])
        setActiveId(target.id)
      }
      const result = await api.ingest(target.id, repoUrl.trim())
      setIngestStatus('queued')
      const poll = async () => {
        try {
          const status = await api.repositoryStatus(result.collection_name)
          setIngestStatus(status.status)
          if (status.status === 'completed' || status.status === 'ready') {
            const updated = await api.getThread(target.id)
            setThread(updated)
            setThreads(await api.listThreads())
            return
          }
          if (status.status === 'failed') { setIngestError(status.error || 'Repository indexing failed.'); return }
          window.setTimeout(poll, 1500)
        } catch (err) {
          setIngestStatus('failed')
          setIngestError(err.message)
        }
      }
      window.setTimeout(poll, 700)
    } catch (err) {
      setIngestStatus('failed')
      setIngestError(err.message)
    }
  }

  const askQuestion = async (event) => {
    event.preventDefault()
    const prompt = question.trim()
    if (!prompt || !activeId || streaming) return
    setQuestion('')
    setThread((current) => current ? {
      ...current,
      messages: [...(current.messages || []), { role: 'user', content: prompt, created_at: new Date().toISOString() }],
    } : current)
    setStreaming({ content: '' })
    try {
      const response = await api.streamChat(activeId, prompt)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''
        events.forEach((event) => {
          const data = event.split('\n').find((line) => line.startsWith('data:'))?.slice(5).trim()
          if (!data) return
          const payload = JSON.parse(data)
          setStreaming((current) => ({ content: `${current?.content || ''}${payload.content || ''}` }))
        })
      }
      setStreaming(null)
      api.getThread(activeId).then(setThread).catch((err) => setError(err.message))
    } catch (err) {
      setStreaming(null)
      setError(err.message)
    }
  }

  const authenticate = (session) => {
    localStorage.setItem('repomind_token', session.access_token)
    setUser(session.user)
  }

  const logout = async () => {
    try { await api.logout() } catch { }
    localStorage.removeItem('repomind_token')
    setUser(null)
    setThread(null)
    setThreads([])
  }

  if (authLoading) return <div className="auth-loading"><LoaderCircle className="spin" size={20} /></div>
  if (!user) return <AuthScreen onAuthenticated={authenticate} />

  return (
    <div className="app-frame">
      <button className="mobile-menu icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open conversations"><Menu size={20} /></button>
      <div className={`sidebar-layer ${sidebarOpen ? 'sidebar-layer-open' : ''}`} onClick={() => setSidebarOpen(false)} />
      <div className={`sidebar-drawer ${sidebarOpen ? 'sidebar-drawer-open' : ''}`}><Sidebar threads={threads} activeId={activeId} onSelect={setActiveId} onNew={createNewThread} onDelete={deleteCurrentThread} disabled={busy} /><button className="mobile-close icon-button" onClick={() => setSidebarOpen(false)}><X size={19} /></button></div>
      <div className="desktop-sidebar"><Sidebar threads={threads} activeId={activeId} onSelect={setActiveId} onNew={createNewThread} onDelete={deleteCurrentThread} disabled={busy} /></div>
      <section className="workspace">
        <div className="workspace-topbar"><div className="topbar-label">LOCAL / RAG WORKSPACE</div><div className="topbar-actions">{error && <span className="error-inline"><AlertCircle size={14} /> {error}</span>}<span className="user-pill">{user.email}</span><button className="logout-button" onClick={logout} title="Log out"><LogOut size={14} /></button></div></div>
        <div className="workspace-content">
          <IngestForm url={repoUrl} setUrl={setRepoUrl} onSubmit={ingest} status={ingestStatus} error={ingestError} disabled={busy} />
          <ChatWindow thread={thread} question={question} setQuestion={setQuestion} onSubmit={askQuestion} streaming={streaming} disabled={busy} />
        </div>
      </section>
    </div>
  )
}