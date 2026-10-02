import { Bot, UserRound } from 'lucide-react'
import MarkdownRenderer from './MarkdownRenderer'

export default function MessageBubble({ message, streaming = false }) {
  const isUser = message.role === 'user'
  return (
    <article className={`message-row ${isUser ? 'message-row-user' : ''}`}>
      <div className={`message-avatar ${isUser ? 'message-avatar-user' : ''}`} aria-hidden="true">
        {isUser ? <UserRound size={15} /> : <Bot size={16} />}
      </div>
      <div className="message-content">
        <div className="message-meta">{isUser ? 'You' : 'RepoMind'}{streaming && <span className="streaming-dot" />}</div>
        <div className={`message-bubble ${isUser ? 'message-bubble-user' : ''}`}>
          {isUser ? <p>{message.content}</p> : <MarkdownRenderer content={message.content || 'Thinking...'} />}
        </div>
        {!isUser && message.sources?.length > 0 && (
          <div className="source-list">
            {message.sources.map((source) => <span className="source-chip" key={source}>{source}</span>)}
          </div>
        )}
      </div>
    </article>
  )
}