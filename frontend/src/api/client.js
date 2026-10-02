const request = async (path, options = {}) => {
  const token = localStorage.getItem('repomind_token')
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}), ...options.headers },
    ...options,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  if (response.status === 204) return null
  return response.json()
}

export const api = {
  signup: (email, password) => request('/api/auth/signup', { method: 'POST', body: JSON.stringify({ email, password }) }),
  login: (email, password) => request('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  me: () => request('/api/auth/me'),
  logout: () => request('/api/auth/logout', { method: 'POST' }),
  listThreads: () => request('/api/threads'),
  getThread: (id) => request(`/api/threads/${id}`),
  createThread: (title = 'New Repository') => request('/api/threads', {
    method: 'POST',
    body: JSON.stringify({ title }),
  }),
  deleteThread: (id) => request(`/api/threads/${id}`, { method: 'DELETE' }),
  ingest: (threadId, githubUrl) => request('/api/repos/ingest', {
    method: 'POST',
    body: JSON.stringify({ thread_id: threadId, github_url: githubUrl }),
  }),
  repositoryStatus: (collectionName) => request(`/api/repos/${collectionName}/status`),
  streamUrl: (threadId, question) => `/api/threads/${threadId}/chat/stream?question=${encodeURIComponent(question)}`,
  streamChat: async (threadId, question) => {
    const token = localStorage.getItem('repomind_token')
    const response = await fetch(`/api/threads/${threadId}/chat/stream?question=${encodeURIComponent(question)}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(body.detail || `Request failed (${response.status})`)
    }
    return response
  },
}