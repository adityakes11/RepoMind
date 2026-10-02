import { Clock3, MessageSquarePlus, Trash2 } from 'lucide-react'

export default function Sidebar({ threads, activeId, onSelect, onNew, onDelete, disabled }) {
  return (
    <aside className="sidebar">
      <div className="brand-lockup">
        <div className="brand-mark">R</div>
        <div><strong>RepoMind</strong><span>codebase intelligence</span></div>
      </div>
      <button className="new-thread-button" onClick={onNew} disabled={disabled}>
        <MessageSquarePlus size={17} /> New conversation
      </button>
      <div className="sidebar-label"><Clock3 size={14} /> Conversations</div>
      <nav className="thread-list" aria-label="Conversations">
        {threads.length === 0 && <p className="empty-sidebar">Your indexed repositories will appear here.</p>}
        {threads.map((thread) => (
          <div className={`thread-item ${thread.id === activeId ? 'thread-item-active' : ''}`} key={thread.id}>
            <button onClick={() => onSelect(thread.id)} className="thread-select">
              <span className="thread-title">{thread.title || 'New conversation'}</span>
              <span className="thread-repo">{thread.repo_name || 'Repository not connected'}</span>
            </button>
            <button className="icon-button thread-delete" title="Delete conversation" onClick={() => onDelete(thread.id)}>
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </nav>
      <div className="sidebar-footer"><span className="status-pip" /> Local workspace</div>
    </aside>
  )
}