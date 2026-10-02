import { ArrowUp, BookOpen, LoaderCircle, Sparkles } from 'lucide-react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ thread, question, setQuestion, onSubmit, streaming, disabled }) {
  const hasRepo = Boolean(thread?.collection_name)
  return (
    <main className="chat-shell">
      <header className="chat-header">
        <div><div className="eyebrow">Workspace / Conversation</div><h1>{thread?.repo_name || 'New conversation'}</h1></div>
        {hasRepo && <div className="repo-badge"><span className="status-pip" /> {thread.repo_name}</div>}
      </header>
      <div className="chat-scroll">
        {!hasRepo && (!thread?.messages?.length) && (
          <div className="welcome-state">
            <div className="welcome-icon"><Sparkles size={23} /></div>
            <div className="eyebrow">Repository intelligence</div>
            <h2>Ask better questions<br /><em>of your code.</em></h2>
            <p>Connect a GitHub repository above, then explore architecture, behavior, and implementation details in plain language.</p>
            <div className="welcome-note"><BookOpen size={15} /> Answers stay grounded in your indexed source.</div>
          </div>
        )}
        {thread?.messages?.map((message, index) => <MessageBubble key={`${message.created_at}-${index}`} message={message} />)}
        {streaming && <MessageBubble message={{ role: 'assistant', content: streaming.content, sources: [] }} streaming />}
      </div>
      <form className="chat-composer" onSubmit={onSubmit}>
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); onSubmit(event) } }} placeholder={hasRepo ? 'Ask about this repository...' : 'Connect a repository to start asking questions'} disabled={disabled || !hasRepo} rows="1" />
        <button className="send-button" type="submit" disabled={disabled || !hasRepo || !question.trim()} title="Send question"><ArrowUp size={18} /></button>
        {streaming && <LoaderCircle className="composer-loader spin" size={16} />}
      </form>
      <div className="composer-hint">Enter to send <span>·</span> Shift + Enter for a new line</div>
    </main>
  )
}