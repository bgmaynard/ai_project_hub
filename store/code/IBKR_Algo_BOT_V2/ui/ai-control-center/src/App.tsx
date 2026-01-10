/**
 * AI Control Center - Full Governor Panel
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import Governor from './components/Governor';

// Error Boundary to catch render errors
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          backgroundColor: '#1e1e1e',
          color: '#f48771',
          padding: '20px',
          minHeight: '100vh',
          fontFamily: 'Segoe UI, Arial, sans-serif'
        }}>
          <h1>AI Control Center - Error</h1>
          <p>Something went wrong loading the Governor panel.</p>
          <pre style={{
            backgroundColor: '#252526',
            padding: '15px',
            borderRadius: '4px',
            overflow: 'auto',
            color: '#d4d4d4'
          }}>
            {this.state.error?.message}
            {'\n\n'}
            {this.state.error?.stack}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: '#007acc',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

function App() {
  return (
    <ErrorBoundary>
      <Governor />
    </ErrorBoundary>
  );
}

export default App;
