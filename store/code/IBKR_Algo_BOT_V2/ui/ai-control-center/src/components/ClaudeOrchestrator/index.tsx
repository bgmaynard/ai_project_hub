import React, { useState, useEffect, useRef } from 'react';
import apiService from '../../services/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

const SUGGESTED_PROMPTS = [
  {
    title: "Analyze Today's Performance",
    prompt: "Analyze my trading performance for today. What patterns do you see and what recommendations do you have?",
    icon: '📊'
  },
  {
    title: 'Strategy Optimization',
    prompt: "Review my current trading strategies and suggest optimizations based on recent performance data.",
    icon: '🎯'
  },
  {
    title: 'Risk Assessment',
    prompt: "Assess the current risk levels across my portfolio and recommend any adjustments.",
    icon: '⚠️'
  },
  {
    title: 'Market Insights',
    prompt: "What are the key market trends I should be aware of based on recent data?",
    icon: '🔍'
  },
  {
    title: 'Model Performance',
    prompt: "Compare the performance of my AI models and identify which ones are performing best.",
    icon: '🤖'
  },
  {
    title: 'Backtest Analysis',
    prompt: "Analyze my recent backtest results and provide insights on potential improvements.",
    icon: '📈'
  }
];

export const ClaudeOrchestrator: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [showConversations, setShowConversations] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConversations();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadConversations = async () => {
    try {
      const response = await apiService.getConversations();
      if (response.success && response.data) {
        setConversations(response.data);
      }
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (conversationId: string) => {
    try {
      const response = await apiService.getConversation(conversationId);
      if (response.success && response.data) {
        setMessages(response.data.messages || []);
        setCurrentConversationId(conversationId);
        setShowConversations(false);
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const sendMessage = async (messageText: string) => {
    if (!messageText.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageText,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await apiService.claudeQuery(messageText, {
        conversation_id: currentConversationId
      });

      if (response.success && response.data) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Update conversation ID if it's a new conversation
        if (!currentConversationId && response.data.conversation_id) {
          setCurrentConversationId(response.data.conversation_id);
          loadConversations();
        }
      }
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message || 'Failed to get response from Claude'}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedPrompt = (prompt: string) => {
    setInputMessage(prompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputMessage);
    }
  };

  const newConversation = () => {
    setMessages([]);
    setCurrentConversationId(null);
    setShowConversations(false);
  };

  const exportConversation = () => {
    const conversationText = messages
      .map(msg => `[${msg.role.toUpperCase()}] ${new Date(msg.timestamp).toLocaleString()}\n${msg.content}\n`)
      .join('\n---\n\n');

    const blob = new Blob([conversationText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `claude-conversation-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="h-full flex">
      {/* Conversation Sidebar */}
      <div className={`${showConversations ? 'w-64' : 'w-0'} bg-ibkr-surface border-r border-ibkr-border transition-all duration-300 overflow-hidden`}>
        <div className="p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-3">Conversations</h3>
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {conversations.map(conv => (
              <button
                key={conv.id}
                onClick={() => loadConversation(conv.id)}
                className={`w-full text-left p-2 rounded text-sm transition-colors ${
                  currentConversationId === conv.id
                    ? 'bg-ibkr-accent text-white'
                    : 'bg-ibkr-bg text-ibkr-text hover:bg-ibkr-border'
                }`}
              >
                <div className="font-medium truncate">{conv.title}</div>
                <div className="text-xs opacity-70 mt-1">
                  {conv.message_count} messages • {new Date(conv.updated_at).toLocaleDateString()}
                </div>
              </button>
            ))}
            {conversations.length === 0 && (
              <p className="text-xs text-ibkr-text-secondary text-center py-4">
                No conversations yet
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full">
        {/* Header */}
        <div className="bg-ibkr-surface border-b border-ibkr-border p-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowConversations(!showConversations)}
              className="p-2 hover:bg-ibkr-bg rounded transition-colors"
              title="Toggle Conversations"
            >
              <svg className="w-5 h-5 text-ibkr-text" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-xl font-bold text-ibkr-text">Claude AI Orchestrator</h1>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={newConversation}
              className="px-3 py-1.5 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text text-sm rounded transition-colors"
            >
              + New Chat
            </button>
            {messages.length > 0 && (
              <button
                onClick={exportConversation}
                className="px-3 py-1.5 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text text-sm rounded transition-colors"
                title="Export Conversation"
              >
                📥 Export
              </button>
            )}
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-8">
                <div className="text-6xl mb-4">🤖</div>
                <h2 className="text-2xl font-bold text-ibkr-text mb-2">
                  Claude AI Trading Assistant
                </h2>
                <p className="text-ibkr-text-secondary">
                  Ask me anything about your trading strategies, performance, or market insights
                </p>
              </div>

              {/* Suggested Prompts */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {SUGGESTED_PROMPTS.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestedPrompt(prompt.prompt)}
                    className="p-4 bg-ibkr-surface border border-ibkr-border rounded-lg hover:border-ibkr-accent transition-colors text-left group"
                  >
                    <div className="text-2xl mb-2">{prompt.icon}</div>
                    <div className="text-sm font-medium text-ibkr-text group-hover:text-ibkr-accent transition-colors">
                      {prompt.title}
                    </div>
                    <div className="text-xs text-ibkr-text-secondary mt-1 line-clamp-2">
                      {prompt.prompt}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl px-4 py-3 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-ibkr-accent text-white'
                    : 'bg-ibkr-surface border border-ibkr-border text-ibkr-text'
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <span className="text-xs opacity-70">
                    {message.role === 'user' ? 'You' : 'Claude'}
                  </span>
                  <span className="text-xs opacity-50">
                    {formatTimestamp(message.timestamp)}
                  </span>
                </div>
                <div className="whitespace-pre-wrap">{message.content}</div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-3xl px-4 py-3 rounded-lg bg-ibkr-surface border border-ibkr-border">
                <div className="flex items-center space-x-2 text-ibkr-text-secondary">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-ibkr-text-secondary rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-ibkr-text-secondary rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-ibkr-text-secondary rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-sm">Claude is thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-ibkr-border bg-ibkr-surface p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex space-x-3">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask Claude anything about your trading strategies, performance, or market insights..."
                className="flex-1 bg-ibkr-bg border border-ibkr-border rounded-lg px-4 py-3 text-ibkr-text placeholder-ibkr-text-secondary focus:outline-none focus:ring-2 focus:ring-ibkr-accent resize-none"
                rows={3}
                disabled={isLoading}
              />
              <button
                onClick={() => sendMessage(inputMessage)}
                disabled={!inputMessage.trim() || isLoading}
                className="px-6 py-3 bg-ibkr-accent hover:bg-opacity-90 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                Send
              </button>
            </div>
            <p className="text-xs text-ibkr-text-secondary mt-2 text-center">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClaudeOrchestrator;
