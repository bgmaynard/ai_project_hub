import { useState, useEffect } from 'react';
import { Activity, TrendingUp, DollarSign, List, Search, Terminal, Brain, BarChart3, Play, Square, Zap } from 'lucide-react';

function App() {
  // State
  const [activeTab, setActiveTab] = useState('overview');
  const [status, setStatus] = useState({
    mtf_running: false,
    warrior_running: false,
    ibkr_connected: false
  });
  
  const [positions, setPositions] = useState([]);
  const [orders, setOrders] = useState([]);
  const [watchlists, setWatchlists] = useState([]);
  const [currentWatchlist, setCurrentWatchlist] = useState(null);
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [scannerResults, setScannerResults] = useState([]);
  const [logs, setLogs] = useState([]);
  
  // NEW: Training & Backtesting State
  const [trainings, setTrainings] = useState({ active: [], completed: [] });
  const [backtests, setBacktests] = useState({ active: [], completed: [] });
  const [trainingConfig, setTrainingConfig] = useState({
    period: '2y',
    interval: '1h',
    epochs: 50,
    batch_size: 32
  });

  // Fetch data
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [statusRes, watchlistsRes, logsRes, trainingsRes, backtestsRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/watchlists'),
        fetch('/api/logs'),
        fetch('/api/train/status'),
        fetch('/api/backtest/status')
      ]);
      
      setStatus(await statusRes.json());
      const wl = await watchlistsRes.json();
      setWatchlists(wl);
      if (!currentWatchlist && wl.length > 0) {
        setCurrentWatchlist(wl[0]);
      }
      setLogs(await logsRes.json());
      setTrainings(await trainingsRes.json());
      setBacktests(await backtestsRes.json());
      
      if (status.ibkr_connected) {
        const [posRes, ordRes] = await Promise.all([
          fetch('/api/ibkr/positions'),
          fetch('/api/ibkr/orders')
        ]);
        setPositions(await posRes.json());
        setOrders(await ordRes.json());
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  // Training functions
  const startTraining = async () => {
    if (selectedSymbols.length === 0) {
      alert('Please select symbols to train');
      return;
    }
    
    try {
      const response = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols: selectedSymbols,
          config: trainingConfig
        })
      });
      
      const result = await response.json();
      if (result.success) {
        alert(`Training started: ${result.training_id}`);
        setSelectedSymbols([]);
        fetchData();
      }
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training');
    }
  };

  const executeTradeFromBacktest = async (symbol, backtestResult) => {
    const quantity = prompt(`How many shares of ${symbol} would you like to buy?`, '100');
    if (!quantity) return;
    
    const parsedQty = parseInt(quantity);
    if (isNaN(parsedQty) || parsedQty <= 0) {
      alert('Please enter a valid quantity');
      return;
    }
    
    try {
      const response = await fetch('/api/trade/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol,
          action: 'BUY',
          quantity: parsedQty,
          order_type: 'MKT'
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        alert(`✅ Order Placed Successfully!\n\n` +
              `Symbol: ${symbol}\n` +
              `Action: BUY\n` +
              `Quantity: ${parsedQty} shares\n` +
              `Order ID: ${result.order_id}\n\n` +
              `Check the Orders tab to see your order.`);
        fetchData(); // Refresh to show new order
      } else {
        alert(`❌ Order Failed\n\n${result.message}`);
      }
    } catch (error) {
      console.error('Error executing trade:', error);
      alert('❌ Failed to execute trade. Check console for details.');
    }
  };

  // Symbol selection
  const toggleSymbolSelection = (symbol) => {
    setSelectedSymbols(prev => 
      prev.includes(symbol) 
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  // Render functions
  const renderOverview = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>System Overview</h2>
      
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '30px'}}>
        <div style={{padding: '20px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #e9ecef'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px'}}>
            <DollarSign size={24} color="#10b981" />
            <h3 style={{fontSize: '16px', fontWeight: '600'}}>Positions</h3>
          </div>
          <p style={{fontSize: '32px', fontWeight: 'bold', color: '#10b981'}}>{positions.length}</p>
          <p style={{fontSize: '14px', color: '#6c757d'}}>Active positions</p>
        </div>

        <div style={{padding: '20px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #e9ecef'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px'}}>
            <List size={24} color="#3b82f6" />
            <h3 style={{fontSize: '16px', fontWeight: '600'}}>Watchlist</h3>
          </div>
          <p style={{fontSize: '32px', fontWeight: 'bold', color: '#3b82f6'}}>
            {currentWatchlist?.symbols?.length || 0}
          </p>
          <p style={{fontSize: '14px', color: '#6c757d'}}>Monitored symbols</p>
        </div>

        <div style={{padding: '20px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #e9ecef'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px'}}>
            <Brain size={24} color="#8b5cf6" />
            <h3 style={{fontSize: '16px', fontWeight: '600'}}>Training</h3>
          </div>
          <p style={{fontSize: '32px', fontWeight: 'bold', color: '#8b5cf6'}}>
            {trainings.active.length}
          </p>
          <p style={{fontSize: '14px', color: '#6c757d'}}>Active trainings</p>
        </div>

        <div style={{padding: '20px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #e9ecef'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px'}}>
            <BarChart3 size={24} color="#f59e0b" />
            <h3 style={{fontSize: '16px', fontWeight: '600'}}>Backtests</h3>
          </div>
          <p style={{fontSize: '32px', fontWeight: 'bold', color: '#f59e0b'}}>
            {backtests.completed.length}
          </p>
          <p style={{fontSize: '14px', color: '#6c757d'}}>Completed tests</p>
        </div>
      </div>

      {trainings.active.length > 0 && (
        <div style={{marginBottom: '30px'}}>
          <h3 style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '15px'}}>
            🔥 Active Trainings
          </h3>
          {trainings.active.map(training => (
            <div key={training.id} style={{
              padding: '15px',
              background: '#fff',
              border: '2px solid #8b5cf6',
              borderRadius: '8px',
              marginBottom: '10px'
            }}>
              <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '10px'}}>
                <span style={{fontWeight: 'bold'}}>
                  {training.current_symbol || training.symbols.join(', ')}
                </span>
                <span style={{color: '#8b5cf6'}}>{training.progress}%</span>
              </div>
              <div style={{
                width: '100%',
                height: '8px',
                background: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${training.progress}%`,
                  height: '100%',
                  background: '#8b5cf6',
                  transition: 'width 0.3s'
                }} />
              </div>
              <div style={{marginTop: '10px', fontSize: '14px', color: '#6c757d'}}>
                Status: {training.status} | Symbols: {training.symbols.length}
              </div>
            </div>
          ))}
        </div>
      )}

      {backtests.completed.length > 0 && (
        <div>
          <h3 style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '15px'}}>
            📊 Recent Backtest Results
          </h3>
          <div style={{overflowX: 'auto'}}>
            <table style={{width: '100%', borderCollapse: 'collapse'}}>
              <thead>
                <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
                  <th style={{padding: '12px', textAlign: 'left'}}>Symbol</th>
                  <th style={{padding: '12px', textAlign: 'right'}}>Return %</th>
                  <th style={{padding: '12px', textAlign: 'right'}}>Win Rate</th>
                  <th style={{padding: '12px', textAlign: 'right'}}>Sharpe</th>
                  <th style={{padding: '12px', textAlign: 'right'}}>Max DD</th>
                  <th style={{padding: '12px', textAlign: 'center'}}>Action</th>
                </tr>
              </thead>
              <tbody>
                {backtests.completed.slice(-5).reverse().map(backtest => 
                  backtest.symbols.map(symbol => {
                    const result = backtest.results?.[symbol] || {};
                    return (
                      <tr key={`${backtest.id}-${symbol}`} style={{borderBottom: '1px solid #e9ecef'}}>
                        <td style={{padding: '12px', fontWeight: 'bold'}}>{symbol}</td>
                        <td style={{
                          padding: '12px',
                          textAlign: 'right',
                          color: result.total_return > 0 ? '#10b981' : '#ef4444',
                          fontWeight: 'bold'
                        }}>
                          {result.total_return?.toFixed(2)}%
                        </td>
                        <td style={{padding: '12px', textAlign: 'right'}}>
                          {result.win_rate?.toFixed(1)}%
                        </td>
                        <td style={{padding: '12px', textAlign: 'right'}}>
                          {result.sharpe_ratio?.toFixed(2)}
                        </td>
                        <td style={{
                          padding: '12px',
                          textAlign: 'right',
                          color: '#ef4444'
                        }}>
                          {result.max_drawdown?.toFixed(2)}%
                        </td>
                        <td style={{padding: '12px', textAlign: 'center'}}>
                          <button
                            onClick={() => executeTradeFromBacktest(symbol, result)}
                            style={{
                              padding: '6px 12px',
                              background: '#10b981',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '14px'
                            }}
                          >
                            Trade
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );

  const renderWatchlist = () => (
    <div>
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px'}}>
        <h2 style={{fontSize: '24px', fontWeight: 'bold'}}>Watchlist Manager</h2>
        
        <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
          {/* Manual Symbol Entry */}
          <input
            type="text"
            placeholder="Enter symbol (e.g., TSLA)"
            style={{
              padding: '8px 12px',
              border: '1px solid #dee2e6',
              borderRadius: '6px',
              fontSize: '14px',
              width: '180px'
            }}
            onKeyPress={async (e) => {
              if (e.key === 'Enter') {
                const symbol = e.target.value.toUpperCase().trim();
                if (symbol) {
                  try {
                    const res = await fetch(`/api/watchlist/${currentWatchlist.name}/add-manual`, {
                      method: 'POST',
                      headers: {'Content-Type': 'application/json'},
                      body: JSON.stringify({symbol})
                    });
                    const data = await res.json();
                    if (data.success) {
                      alert(`Added ${symbol} to ${currentWatchlist.name}`);
                      e.target.value = '';
                      fetchData();
                    } else {
                      alert(data.message || 'Failed to add symbol');
                    }
                  } catch (error) {
                    alert('Error adding symbol');
                  }
                }
              }
            }}
          />
          
          <select 
            value={currentWatchlist?.name || ''}
            onChange={(e) => {
              const wl = watchlists.find(w => w.name === e.target.value);
              setCurrentWatchlist(wl);
              setSelectedSymbols([]);
            }}
            style={{
              padding: '8px 12px',
              border: '1px solid #dee2e6',
              borderRadius: '6px',
              fontSize: '14px'
            }}
          >
            {watchlists.map(wl => (
              <option key={wl.name} value={wl.name}>{wl.name}</option>
            ))}
          </select>

          {selectedSymbols.length > 0 && (
            <button
              onClick={startTraining}
              style={{
                padding: '8px 16px',
                background: '#8b5cf6',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontWeight: '600'
              }}
            >
              <Brain size={18} />
              Train Selected ({selectedSymbols.length})
            </button>
          )}
        </div>
      </div>

      {/* Helper text for manual entry */}
      <div style={{
        marginBottom: '15px',
        padding: '10px',
        background: '#f0f9ff',
        borderRadius: '6px',
        fontSize: '14px',
        color: '#0369a1'
      }}>
        💡 <strong>Tip:</strong> Type a symbol in the input field and press Enter to add it manually, or use the Scanner tab to find stocks.
      </div>

      {selectedSymbols.length > 0 && (
        <div style={{
          padding: '15px',
          background: '#f8f9fa',
          borderRadius: '8px',
          marginBottom: '20px',
          border: '1px solid #dee2e6'
        }}>
          <h3 style={{fontSize: '16px', fontWeight: 'bold', marginBottom: '10px'}}>
            Training Configuration
          </h3>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px'}}>
            <div>
              <label style={{fontSize: '12px', color: '#6c757d', marginBottom: '5px', display: 'block'}}>
                Period
              </label>
              <select
                value={trainingConfig.period}
                onChange={(e) => setTrainingConfig({...trainingConfig, period: e.target.value})}
                style={{width: '100%', padding: '6px', border: '1px solid #dee2e6', borderRadius: '4px'}}
              >
                <option value="1y">1 Year</option>
                <option value="2y">2 Years</option>
                <option value="5y">5 Years</option>
              </select>
            </div>
            <div>
              <label style={{fontSize: '12px', color: '#6c757d', marginBottom: '5px', display: 'block'}}>
                Interval
              </label>
              <select
                value={trainingConfig.interval}
                onChange={(e) => setTrainingConfig({...trainingConfig, interval: e.target.value})}
                style={{width: '100%', padding: '6px', border: '1px solid #dee2e6', borderRadius: '4px'}}
              >
                <option value="1h">1 Hour</option>
                <option value="1d">1 Day</option>
                <option value="1wk">1 Week</option>
              </select>
            </div>
            <div>
              <label style={{fontSize: '12px', color: '#6c757d', marginBottom: '5px', display: 'block'}}>
                Epochs
              </label>
              <input
                type="number"
                value={trainingConfig.epochs}
                onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
                style={{width: '100%', padding: '6px', border: '1px solid #dee2e6', borderRadius: '4px'}}
              />
            </div>
            <div>
              <label style={{fontSize: '12px', color: '#6c757d', marginBottom: '5px', display: 'block'}}>
                Batch Size
              </label>
              <input
                type="number"
                value={trainingConfig.batch_size}
                onChange={(e) => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
                style={{width: '100%', padding: '6px', border: '1px solid #dee2e6', borderRadius: '4px'}}
              />
            </div>
          </div>
        </div>
      )}

      <div style={{
        background: 'white',
        border: '1px solid #dee2e6',
        borderRadius: '8px',
        overflow: 'hidden'
      }}>
        <table style={{width: '100%', borderCollapse: 'collapse'}}>
          <thead>
            <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
              <th style={{padding: '12px', textAlign: 'left', width: '50px'}}>
                <input
                  type="checkbox"
                  checked={selectedSymbols.length === currentWatchlist?.symbols?.length && currentWatchlist?.symbols?.length > 0}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedSymbols(currentWatchlist?.symbols || []);
                    } else {
                      setSelectedSymbols([]);
                    }
                  }}
                  style={{cursor: 'pointer'}}
                />
              </th>
              <th style={{padding: '12px', textAlign: 'left'}}>Symbol</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Last Price</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Change</th>
              <th style={{padding: '12px', textAlign: 'center'}}>Action</th>
            </tr>
          </thead>
          <tbody>
            {currentWatchlist?.symbols?.map(symbol => (
              <tr key={symbol} style={{borderBottom: '1px solid #e9ecef'}}>
                <td style={{padding: '12px'}}>
                  <input
                    type="checkbox"
                    checked={selectedSymbols.includes(symbol)}
                    onChange={() => toggleSymbolSelection(symbol)}
                    style={{cursor: 'pointer'}}
                  />
                </td>
                <td style={{padding: '12px', fontWeight: 'bold'}}>{symbol}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>$--</td>
                <td style={{padding: '12px', textAlign: 'right', color: '#10b981'}}>+0.00%</td>
                <td style={{padding: '12px', textAlign: 'center'}}>
                  <button
                    onClick={async () => {
                      if (confirm(`Remove ${symbol} from watchlist?`)) {
                        await fetch(`/api/watchlist/${currentWatchlist.name}/remove`, {
                          method: 'POST',
                          headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify({symbol})
                        });
                        fetchData();
                      }
                    }}
                    style={{
                      padding: '4px 8px',
                      background: '#ef4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {currentWatchlist?.symbols?.length === 0 && (
        <div style={{
          padding: '40px',
          textAlign: 'center',
          color: '#6c757d'
        }}>
          No symbols in this watchlist. Add symbols from the Scanner tab.
        </div>
      )}
    </div>
  );

  const renderTraining = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Training Monitor</h2>

      <div style={{marginBottom: '30px'}}>
        <h3 style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '15px'}}>
          Active Trainings ({trainings.active.length})
        </h3>
        
        {trainings.active.length === 0 ? (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            background: '#f8f9fa',
            borderRadius: '8px',
            color: '#6c757d'
          }}>
            <Brain size={48} style={{margin: '0 auto 15px'}} />
            <p>No active trainings. Select symbols from the Watchlist tab to start training.</p>
          </div>
        ) : (
          trainings.active.map(training => (
            <div key={training.id} style={{
              padding: '20px',
              background: 'white',
              border: '2px solid #8b5cf6',
              borderRadius: '8px',
              marginBottom: '15px'
            }}>
              <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '15px'}}>
                <div>
                  <h4 style={{fontSize: '16px', fontWeight: 'bold', marginBottom: '5px'}}>
                    Training ID: {training.id}
                  </h4>
                  <p style={{fontSize: '14px', color: '#6c757d'}}>
                    Symbols: {training.symbols.join(', ')}
                  </p>
                </div>
                <div style={{textAlign: 'right'}}>
                  <div style={{fontSize: '24px', fontWeight: 'bold', color: '#8b5cf6'}}>
                    {training.progress}%
                  </div>
                  <div style={{fontSize: '12px', color: '#6c757d', textTransform: 'uppercase'}}>
                    {training.status}
                  </div>
                </div>
              </div>

              <div style={{
                width: '100%',
                height: '12px',
                background: '#e9ecef',
                borderRadius: '6px',
                overflow: 'hidden',
                marginBottom: '15px'
              }}>
                <div style={{
                  width: `${training.progress}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
                  transition: 'width 0.5s ease'
                }} />
              </div>

              {training.current_symbol && (
                <div style={{
                  padding: '10px',
                  background: '#f8f9fa',
                  borderRadius: '6px',
                  marginBottom: '15px'
                }}>
                  <span style={{fontSize: '14px', color: '#6c757d'}}>Currently training: </span>
                  <span style={{fontSize: '14px', fontWeight: 'bold'}}>{training.current_symbol}</span>
                </div>
              )}

              {training.logs && training.logs.length > 0 && (
                <div style={{
                  maxHeight: '150px',
                  overflowY: 'auto',
                  background: '#f8f9fa',
                  padding: '10px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontFamily: 'monospace'
                }}>
                  {training.logs.slice(-10).map((log, idx) => (
                    <div key={idx} style={{marginBottom: '5px'}}>
                      <span style={{color: '#6c757d'}}>{new Date(log.timestamp).toLocaleTimeString()}</span>
                      {' - '}
                      <span>{log.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      <div>
        <h3 style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '15px'}}>
          Recent Completions ({trainings.completed.length})
        </h3>

        {trainings.completed.length === 0 ? (
          <div style={{
            padding: '20px',
            background: '#f8f9fa',
            borderRadius: '8px',
            textAlign: 'center',
            color: '#6c757d'
          }}>
            No completed trainings yet.
          </div>
        ) : (
          <div style={{overflowX: 'auto'}}>
            <table style={{width: '100%', borderCollapse: 'collapse', background: 'white'}}>
              <thead>
                <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
                  <th style={{padding: '12px', textAlign: 'left'}}>Training ID</th>
                  <th style={{padding: '12px', textAlign: 'left'}}>Symbols</th>
                  <th style={{padding: '12px', textAlign: 'center'}}>Status</th>
                  <th style={{padding: '12px', textAlign: 'right'}}>Duration</th>
                  <th style={{padding: '12px', textAlign: 'right'}}>Completed</th>
                </tr>
              </thead>
              <tbody>
                {trainings.completed.map(training => {
                  const duration = training.completed_at && training.started_at
                    ? Math.round((new Date(training.completed_at) - new Date(training.started_at)) / 1000)
                    : 0;
                  
                  return (
                    <tr key={training.id} style={{borderBottom: '1px solid #e9ecef'}}>
                      <td style={{padding: '12px', fontFamily: 'monospace', fontSize: '12px'}}>
                        {training.id}
                      </td>
                      <td style={{padding: '12px'}}>
                        {training.symbols.join(', ')}
                      </td>
                      <td style={{padding: '12px', textAlign: 'center'}}>
                        <span style={{
                          padding: '4px 8px',
                          background: training.status === 'completed' ? '#d1fae5' : '#fee2e2',
                          color: training.status === 'completed' ? '#065f46' : '#991b1b',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '600'
                        }}>
                          {training.status}
                        </span>
                      </td>
                      <td style={{padding: '12px', textAlign: 'right'}}>
                        {duration}s
                      </td>
                      <td style={{padding: '12px', textAlign: 'right', fontSize: '12px', color: '#6c757d'}}>
                        {new Date(training.completed_at).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );

  const renderBacktesting = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Backtest Results</h2>

      {backtests.active.length > 0 && (
        <div style={{marginBottom: '30px'}}>
          <h3 style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '15px'}}>
            Running Backtests
          </h3>
          {backtests.active.map(backtest => (
            <div key={backtest.id} style={{
              padding: '15px',
              background: '#fff3cd',
              border: '2px solid #f59e0b',
              borderRadius: '8px',
              marginBottom: '10px'
            }}>
              <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '10px'}}>
                <span style={{fontWeight: 'bold'}}>
                  {backtest.current_symbol || 'Processing...'}
                </span>
                <span style={{color: '#f59e0b'}}>{backtest.progress}%</span>
              </div>
              <div style={{
                width: '100%',
                height: '8px',
                background: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${backtest.progress}%`,
                  height: '100%',
                  background: '#f59e0b',
                  transition: 'width 0.3s'
                }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {backtests.completed.length === 0 ? (
        <div style={{
          padding: '60px',
          textAlign: 'center',
          background: '#f8f9fa',
          borderRadius: '8px',
          color: '#6c757d'
        }}>
          <BarChart3 size={64} style={{margin: '0 auto 20px'}} />
          <h3 style={{fontSize: '18px', marginBottom: '10px'}}>No Backtest Results Yet</h3>
          <p>Backtests will run automatically after model training completes.</p>
        </div>
      ) : (
        <div style={{overflowX: 'auto'}}>
          <table style={{width: '100%', borderCollapse: 'collapse', background: 'white'}}>
            <thead>
              <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
                <th style={{padding: '12px', textAlign: 'left'}}>Symbol</th>
                <th style={{padding: '12px', textAlign: 'right'}}>Total Return</th>
                <th style={{padding: '12px', textAlign: 'right'}}>Win Rate</th>
                <th style={{padding: '12px', textAlign: 'right'}}>Sharpe Ratio</th>
                <th style={{padding: '12px', textAlign: 'right'}}>Max Drawdown</th>
                <th style={{padding: '12px', textAlign: 'right'}}>Total Trades</th>
                <th style={{padding: '12px', textAlign: 'center'}}>Action</th>
              </tr>
            </thead>
            <tbody>
              {backtests.completed.map(backtest =>
                backtest.symbols.map(symbol => {
                  const result = backtest.results?.[symbol] || {};
                  return (
                    <tr key={`${backtest.id}-${symbol}`} style={{borderBottom: '1px solid #e9ecef'}}>
                      <td style={{padding: '12px', fontWeight: 'bold', fontSize: '16px'}}>
                        {symbol}
                      </td>
                      <td style={{
                        padding: '12px',
                        textAlign: 'right',
                        fontSize: '18px',
                        fontWeight: 'bold',
                        color: result.total_return > 0 ? '#10b981' : '#ef4444'
                      }}>
                        {result.total_return > 0 ? '+' : ''}{result.total_return?.toFixed(2)}%
                      </td>
                      <td style={{
                        padding: '12px',
                        textAlign: 'right',
                        fontSize: '16px',
                        fontWeight: '600'
                      }}>
                        {result.win_rate?.toFixed(1)}%
                      </td>
                      <td style={{padding: '12px', textAlign: 'right', fontSize: '16px'}}>
                        {result.sharpe_ratio?.toFixed(2)}
                      </td>
                      <td style={{
                        padding: '12px',
                        textAlign: 'right',
                        fontSize: '16px',
                        color: '#ef4444',
                        fontWeight: '600'
                      }}>
                        {result.max_drawdown?.toFixed(2)}%
                      </td>
                      <td style={{padding: '12px', textAlign: 'right'}}>
                        {result.total_trades}
                      </td>
                      <td style={{padding: '12px', textAlign: 'center'}}>
                        <button
                          onClick={() => executeTradeFromBacktest(symbol, result)}
                          disabled={!status.ibkr_connected}
                          style={{
                            padding: '8px 16px',
                            background: status.ibkr_connected ? '#10b981' : '#9ca3af',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: status.ibkr_connected ? 'pointer' : 'not-allowed',
                            fontWeight: '600',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            margin: '0 auto'
                          }}
                        >
                          <Zap size={16} />
                          Execute Trade
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  const renderPositions = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Positions</h2>
      {positions.length === 0 ? (
        <div style={{padding: '40px', textAlign: 'center', color: '#6c757d'}}>
          No active positions
        </div>
      ) : (
        <table style={{width: '100%', borderCollapse: 'collapse'}}>
          <thead>
            <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
              <th style={{padding: '12px', textAlign: 'left'}}>Symbol</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Position</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Avg Cost</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Market Value</th>
              <th style={{padding: '12px', textAlign: 'right'}}>P&L</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos, idx) => (
              <tr key={idx} style={{borderBottom: '1px solid #e9ecef'}}>
                <td style={{padding: '12px', fontWeight: 'bold'}}>{pos.symbol}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>{pos.position}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>${pos.avg_cost?.toFixed(2)}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>${pos.market_value?.toFixed(2)}</td>
                <td style={{
                  padding: '12px',
                  textAlign: 'right',
                  color: pos.unrealized_pnl >= 0 ? '#10b981' : '#ef4444',
                  fontWeight: 'bold'
                }}>
                  ${pos.unrealized_pnl?.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );

  const renderOrders = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Open Orders</h2>
      {orders.length === 0 ? (
        <div style={{padding: '40px', textAlign: 'center', color: '#6c757d'}}>
          No open orders
        </div>
      ) : (
        <table style={{width: '100%', borderCollapse: 'collapse'}}>
          <thead>
            <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
              <th style={{padding: '12px', textAlign: 'left'}}>Symbol</th>
              <th style={{padding: '12px', textAlign: 'left'}}>Action</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Quantity</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Price</th>
              <th style={{padding: '12px', textAlign: 'left'}}>Status</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order, idx) => (
              <tr key={idx} style={{borderBottom: '1px solid #e9ecef'}}>
                <td style={{padding: '12px', fontWeight: 'bold'}}>{order.symbol}</td>
                <td style={{padding: '12px'}}>{order.action}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>{order.quantity}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>${order.price}</td>
                <td style={{padding: '12px'}}>{order.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );

  const renderScanner = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Market Scanner</h2>
      <button
        onClick={async () => {
          const res = await fetch('/api/ibkr/scanner');
          const data = await res.json();
          setScannerResults(data.results || []);
        }}
        style={{
          padding: '10px 20px',
          background: '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          cursor: 'pointer',
          marginBottom: '20px',
          fontWeight: '600'
        }}
      >
        Run Scanner
      </button>

      {scannerResults.length > 0 && (
        <table style={{width: '100%', borderCollapse: 'collapse'}}>
          <thead>
            <tr style={{background: '#f8f9fa', borderBottom: '2px solid #dee2e6'}}>
              <th style={{padding: '12px', textAlign: 'left'}}>Symbol</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Price</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Change %</th>
              <th style={{padding: '12px', textAlign: 'right'}}>Volume</th>
              <th style={{padding: '12px', textAlign: 'center'}}>Action</th>
            </tr>
          </thead>
          <tbody>
            {scannerResults.map((result, idx) => (
              <tr key={idx} style={{borderBottom: '1px solid #e9ecef'}}>
                <td style={{padding: '12px', fontWeight: 'bold'}}>{result.symbol}</td>
                <td style={{padding: '12px', textAlign: 'right'}}>${result.price}</td>
                <td style={{padding: '12px', textAlign: 'right', color: '#10b981'}}>
                  +{result.change}%
                </td>
                <td style={{padding: '12px', textAlign: 'right'}}>{result.volume}</td>
                <td style={{padding: '12px', textAlign: 'center'}}>
                  <button
                    onClick={async () => {
                      await fetch(`/api/watchlist/${currentWatchlist.name}/add`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({symbol: result.symbol})
                      });
                      alert(`Added ${result.symbol} to ${currentWatchlist.name}`);
                      fetchData();
                    }}
                    style={{
                      padding: '6px 12px',
                      background: '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '14px'
                    }}
                  >
                    Add to Watchlist
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );

  const renderLogs = () => (
    <div>
      <h2 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Activity Log</h2>
      <div style={{
        background: '#1e1e1e',
        color: '#d4d4d4',
        padding: '15px',
        borderRadius: '8px',
        fontFamily: 'monospace',
        fontSize: '13px',
        maxHeight: '600px',
        overflowY: 'auto'
      }}>
        {logs.length === 0 ? (
          <div style={{color: '#6c757d', textAlign: 'center', padding: '20px'}}>
            No activity yet
          </div>
        ) : (
          logs.map((log, idx) => {
            const colors = {
              error: '#ef4444',
              warning: '#f59e0b',
              success: '#10b981',
              info: '#3b82f6'
            };
            return (
              <div key={idx} style={{marginBottom: '8px'}}>
                <span style={{color: '#6c757d'}}>
                  [{new Date(log.timestamp).toLocaleTimeString()}]
                </span>
                {' '}
                <span style={{color: colors[log.level] || '#d4d4d4'}}>
                  [{log.level.toUpperCase()}]
                </span>
                {' '}
                <span style={{color: '#a78bfa'}}>
                  [{log.category}]
                </span>
                {' '}
                <span>{log.message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );

  return (
    <div style={{minHeight: '100vh', background: '#f8f9fa'}}>
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '20px 40px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{fontSize: '28px', fontWeight: 'bold', marginBottom: '10px'}}>
          🤖 AI Trading Bot Dashboard
        </h1>
        <p style={{fontSize: '14px', opacity: 0.9}}>
          Multi-Timeframe + Warrior Trading System
        </p>
      </div>

      <div style={{padding: '20px 40px'}}>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '20px'}}>
          <div style={{
            background: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            border: `2px solid ${status.mtf_running ? '#10b981' : '#6c757d'}`
          }}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '15px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                <TrendingUp size={24} color={status.mtf_running ? '#10b981' : '#6c757d'} />
                <h3 style={{fontSize: '16px', fontWeight: '600'}}>MTF Bot</h3>
              </div>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: status.mtf_running ? '#10b981' : '#6c757d'
              }} />
            </div>
            <button
              style={{
                width: '100%',
                padding: '8px',
                background: status.mtf_running ? '#ef4444' : '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {status.mtf_running ? <Square size={16} /> : <Play size={16} />}
              {status.mtf_running ? 'Stop' : 'Start'}
            </button>
          </div>

          <div style={{
            background: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            border: `2px solid ${status.warrior_running ? '#f59e0b' : '#6c757d'}`
          }}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '15px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                <Activity size={24} color={status.warrior_running ? '#f59e0b' : '#6c757d'} />
                <h3 style={{fontSize: '16px', fontWeight: '600'}}>Warrior Bot</h3>
              </div>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: status.warrior_running ? '#f59e0b' : '#6c757d'
              }} />
            </div>
            <button
              style={{
                width: '100%',
                padding: '8px',
                background: status.warrior_running ? '#ef4444' : '#f59e0b',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {status.warrior_running ? <Square size={16} /> : <Play size={16} />}
              {status.warrior_running ? 'Stop' : 'Start'}
            </button>
          </div>

          <div style={{
            background: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            border: `2px solid ${status.ibkr_connected ? '#3b82f6' : '#6c757d'}`
          }}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '15px'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                <DollarSign size={24} color={status.ibkr_connected ? '#3b82f6' : '#6c757d'} />
                <h3 style={{fontSize: '16px', fontWeight: '600'}}>IBKR TWS</h3>
              </div>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: status.ibkr_connected ? '#3b82f6' : '#6c757d'
              }} />
            </div>
            <button
              onClick={async () => {
                const endpoint = status.ibkr_connected ? 'disconnect' : 'connect';
                await fetch(`/api/ibkr/${endpoint}`, {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({port: 7497})
                });
                fetchData();
              }}
              style={{
                width: '100%',
                padding: '8px',
                background: status.ibkr_connected ? '#ef4444' : '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              {status.ibkr_connected ? 'Disconnect' : 'Connect'}
            </button>
          </div>
        </div>

        <div style={{
          background: 'white',
          borderRadius: '8px',
          padding: '10px',
          marginBottom: '20px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          display: 'flex',
          gap: '10px',
          flexWrap: 'wrap'
        }}>
          {[
            {id: 'overview', label: 'Overview', icon: Activity},
            {id: 'positions', label: 'Positions', icon: DollarSign},
            {id: 'orders', label: 'Orders', icon: List},
            {id: 'watchlist', label: 'Watchlist', icon: List},
            {id: 'training', label: 'Training', icon: Brain},
            {id: 'backtesting', label: 'Backtesting', icon: BarChart3},
            {id: 'scanner', label: 'Scanner', icon: Search},
            {id: 'logs', label: 'Logs', icon: Terminal}
          ].map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '10px 20px',
                  background: activeTab === tab.id ? '#667eea' : 'transparent',
                  color: activeTab === tab.id ? 'white' : '#6c757d',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  transition: 'all 0.2s'
                }}
              >
                <Icon size={18} />
                {tab.label}
              </button>
            );
          })}
        </div>

        <div style={{
          background: 'white',
          borderRadius: '8px',
          padding: '30px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          minHeight: '500px'
        }}>
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'positions' && renderPositions()}
          {activeTab === 'orders' && renderOrders()}
          {activeTab === 'watchlist' && renderWatchlist()}
          {activeTab === 'training' && renderTraining()}
          {activeTab === 'backtesting' && renderBacktesting()}
          {activeTab === 'scanner' && renderScanner()}
          {activeTab === 'logs' && renderLogs()}
        </div>
      </div>
    </div>
  );
}

export default App;