import { useEffect, useState } from 'react'
import { useSymbolStore } from '../stores/symbolStore'

interface NewsItem {
  id: string
  time: string
  releaseTime: string  // Original release time from source
  symbol: string
  headline: string
  source?: string
  sentiment?: 'positive' | 'negative' | 'neutral'
  urgency?: 'high' | 'medium' | 'low'
}

export default function BreakingNews() {
  const { setActiveSymbol, newsFilterSymbol, setNewsFilter } = useSymbolStore()
  const [news, setNews] = useState<NewsItem[]>([])
  const [localFilter, setLocalFilter] = useState<string>('') // Local text filter

  // Use store filter if set, otherwise use local filter
  const activeFilter = newsFilterSymbol || localFilter

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const response = await fetch('/api/news-log/today')
        const data = await response.json()

        // API returns { success: true, news: [...], count: N }
        const entries = data?.news || data?.entries || (Array.isArray(data) ? data : [])
        if (entries.length > 0) {
          const mapped: NewsItem[] = entries.slice(0, 100).map((n: any, i: number) => {
            // Parse release time from various API formats
            // Priority: published_at (original release) > time_str > time
            let releaseTime = ''
            if (n.published_at) {
              try {
                const date = new Date(n.published_at)
                releaseTime = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              } catch {
                // If it's already a formatted string like "12:26 PM"
                releaseTime = n.published_at
              }
            } else if (n.time_str) {
              // Already formatted time string
              releaseTime = n.time_str
            } else if (n.time) {
              // Fallback to time field - may be formatted already or ISO
              try {
                if (n.time.includes('T') || n.time.includes('-')) {
                  const date = new Date(n.time)
                  releaseTime = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                } else {
                  releaseTime = n.time
                }
              } catch {
                releaseTime = n.time
              }
            } else {
              releaseTime = n.timestamp || ''
            }

            return {
              id: n.id || `news-${i}`,
              time: n.time_str || n.time || n.timestamp || new Date().toLocaleTimeString(),
              releaseTime: releaseTime,
              symbol: n.symbol || '',
              headline: n.headline || n.synopsis || n.title || '',
              source: n.source || '',
              sentiment: n.sentiment || 'neutral',
              urgency: n.urgency || 'medium',
            }
          })
          setNews(mapped)
        }
      } catch (err) {
        console.error('Failed to load news:', err)
      }
    }

    fetchNews()
    const interval = setInterval(fetchNews, 30000) // 30 second refresh
    return () => clearInterval(interval)
  }, [])

  const filteredNews = activeFilter
    ? news.filter((n) => n.symbol.toUpperCase().includes(activeFilter.toUpperCase()))
    : news

  const handleNewsClick = (item: NewsItem) => {
    if (item.symbol) {
      setActiveSymbol(item.symbol)
    }
  }

  const getSentimentColor = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive':
        return 'border-l-up'
      case 'negative':
        return 'border-l-down'
      default:
        return 'border-l-sterling-border'
    }
  }

  const getUrgencyBadge = (urgency?: string) => {
    if (urgency === 'high') {
      return (
        <span className="px-1 py-0.5 bg-down text-white text-xxs rounded font-bold">
          HOT
        </span>
      )
    }
    return null
  }

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border">
        <div className="flex items-center gap-2">
          <span className="font-bold text-sterling-text">BREAKING NEWS</span>
          {newsFilterSymbol && (
            <span className="px-1.5 py-0.5 bg-accent-primary text-white text-xxs rounded flex items-center gap-1">
              ${newsFilterSymbol}
              <button
                onClick={() => setNewsFilter(null)}
                className="hover:text-down ml-1"
                title="Clear filter"
              >
                ✕
              </button>
            </span>
          )}
        </div>
        <input
          type="text"
          value={localFilter}
          onChange={(e) => setLocalFilter(e.target.value)}
          placeholder="Filter..."
          className="w-16 px-1 py-0.5 bg-sterling-bg border border-sterling-border text-xxs rounded"
        />
      </div>

      {/* News Feed */}
      <div className="flex-1 overflow-auto">
        {filteredNews.length === 0 ? (
          <div className="text-center py-4 text-sterling-muted">
            No breaking news
          </div>
        ) : (
          filteredNews.map((item) => (
            <div
              key={item.id}
              onClick={() => handleNewsClick(item)}
              className={`px-2 py-1.5 border-b border-sterling-border border-l-2 ${getSentimentColor(
                item.sentiment
              )} hover:bg-sterling-highlight cursor-pointer`}
            >
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-sterling-muted text-xxs" title="Release time">
                  {item.releaseTime || item.time}
                </span>
                {item.symbol && (
                  <span className="text-accent-primary font-bold">
                    ${item.symbol}
                  </span>
                )}
                {getUrgencyBadge(item.urgency)}
              </div>
              <div className="text-sterling-text line-clamp-2">
                {item.headline}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="px-2 py-1 bg-sterling-header border-t border-sterling-border text-xxs text-sterling-muted text-center">
        {filteredNews.length} items • Auto-refreshes every 30s
      </div>
    </div>
  )
}
