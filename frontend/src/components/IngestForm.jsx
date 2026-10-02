import { Check, Database, GitBranch, LoaderCircle, Link2, X } from 'lucide-react'

export default function IngestForm({ url, setUrl, onSubmit, status, error, disabled }) {
  const isWorking = status === 'queued' || status === 'running'
  return (
    <section className="ingest-panel">
      <div className="panel-heading">
        <div className="panel-icon"><GitBranch size={19} /></div>
        <div><h2>Connect a repository</h2><p>Index source code into a private local knowledge base.</p></div>
      </div>
      <form onSubmit={onSubmit} className="ingest-form">
        <div className="url-input-wrap"><Link2 size={17} /><input value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://github.com/owner/repository" aria-label="GitHub repository URL" disabled={disabled || isWorking} /></div>
        <button className="primary-button" type="submit" disabled={disabled || isWorking || !url.trim()}>
          {isWorking ? <LoaderCircle className="spin" size={16} /> : <Database size={16} />}
          {isWorking ? 'Indexing' : 'Analyze repository'}
        </button>
      </form>
      {(isWorking || status === 'completed' || status === 'failed') && (
        <div className={`ingest-status ingest-status-${status}`}>
          {status === 'completed' && <Check size={15} />}{status === 'failed' && <X size={15} />}{isWorking && <LoaderCircle className="spin" size={15} />}
          <span>{status === 'completed' ? 'Repository ready for questions.' : status === 'failed' ? error : `Indexing repository${status === 'running' ? '...' : '...'}`}</span>
          {isWorking && <span className="progress-value">{status === 'running' ? 'Working' : 'Queued'}</span>}
        </div>
      )}
    </section>
  )
}