import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface WorklistItem {
  symbol: string;
  exchange: string;
  current_price?: number;
  bid?: number;
  ask?: number;
  change?: number;
  change_percent?: number;
  volume?: number;
  prediction?: string;
  confidence?: number;
  predicted_change?: number;
  analysis?: string;
  notes?: string;
  added_at: string;
  has_live_data: boolean;
}

interface ScannerResult {
  rank: number;
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  selected?: boolean;
}

export const Worklist: React.FC = () => {
  const [worklist, setWorklist] = useState<WorklistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showScannerDialog, setShowScannerDialog] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [scannerPresets, setScannerPresets] = useState<any[]>([]);
  const [selectedScanner, setSelectedScanner] = useState('TOP_PERC_GAIN');
  const [scannerResults, setScannerResults] = useState<ScannerResult[]>([]);
  const [scanning, setScanning] = useState(false);
  const [editingNotes, setEditingNotes] = useState<string | null>(null);
  const [notesText, setNotesText] = useState('');

  useEffect(() => {
    loadWorklist();
    loadScannerPresets();

    // Auto-refresh every 5 seconds
    const interval = setInterval(loadWorklist, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadWorklist = async () => {
    try {
      const response = await apiService.getWorklist();
      if (response.success && response.data) {
        setWorklist(response.data);
      }
    } catch (error) {
      console.error('Failed to load worklist:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadScannerPresets = async () => {
    try {
      const response = await apiService.getScannerPresets();
      if (response.success && response.data) {
        setScannerPresets(response.data);
      }
    } catch (error) {
      console.error('Failed to load scanner presets:', error);
    }
  };

  const addSymbol = async () => {
    if (!newSymbol.trim()) return;

    try {
      await apiService.addToWorklist({ symbol: newSymbol.toUpperCase(), exchange: 'SMART' });
      setNewSymbol('');
      setShowAddDialog(false);
      await loadWorklist();
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Failed to add symbol');
    }
  };

  const removeSymbol = async (symbol: string) => {
    if (!window.confirm(`Remove ${symbol} from worklist?`)) return;

    try {
      await apiService.removeFromWorklist(symbol);
      await loadWorklist();
    } catch (error) {
      console.error('Failed to remove symbol:', error);
    }
  };

  const runScanner = async () => {
    setScanning(true);
    try {
      const response = await apiService.runScanner({
        scan_code: selectedScanner,
        instrument: 'STK',
        location: 'STK.US.MAJOR',
        num_rows: 50
      });

      if (response.success && response.data) {
        setScannerResults(
          response.data.results.map((r: any) => ({
            ...r,
            selected: false
          }))
        );
      }
    } catch (error) {
      console.error('Failed to run scanner:', error);
    } finally {
      setScanning(false);
    }
  };

  const toggleScannerSelection = (symbol: string) => {
    setScannerResults(prev =>
      prev.map(r => (r.symbol === symbol ? { ...r, selected: !r.selected } : r))
    );
  };

  const addSelectedToWorklist = async () => {
    const selected = scannerResults.filter(r => r.selected).map(r => r.symbol);

    if (selected.length === 0) {
      alert('No symbols selected');
      return;
    }

    try {
      await apiService.addScannerToWorklist({ symbols: selected });
      setShowScannerDialog(false);
      setScannerResults([]);
      await loadWorklist();
    } catch (error) {
      console.error('Failed to add scanner results:', error);
    }
  };

  const updateNotes = async (symbol: string, notes: string) => {
    try {
      await apiService.updateWorklistNotes(symbol, { notes });
      setEditingNotes(null);
      await loadWorklist();
    } catch (error) {
      console.error('Failed to update notes:', error);
    }
  };

  const clearWorklist = async () => {
    if (!window.confirm('Clear entire worklist?')) return;

    try {
      await apiService.clearWorklist();
      await loadWorklist();
    } catch (error) {
      console.error('Failed to clear worklist:', error);
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-ibkr-text-secondary">Loading worklist...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-ibkr-text">Worklist</h1>
          <p className="text-sm text-ibkr-text-secondary mt-1">
            Shared watchlist with AI predictions across all modules
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowScannerDialog(true)}
            className="px-4 py-2 bg-ibkr-surface border border-ibkr-border hover:bg-ibkr-bg text-ibkr-text rounded transition-colors flex items-center space-x-2"
          >
            <span>=</span>
            <span>IBKR Scanner</span>
          </button>
          <button
            onClick={() => setShowAddDialog(true)}
            className="px-4 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors flex items-center space-x-2"
          >
            <span>+</span>
            <span>Add Symbol</span>
          </button>
          {worklist.length > 0 && (
            <button
              onClick={clearWorklist}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Total Symbols</div>
          <div className="text-2xl font-bold text-ibkr-text">{worklist.length}</div>
        </div>
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">With Live Data</div>
          <div className="text-2xl font-bold text-green-500">
            {worklist.filter(w => w.has_live_data).length}
          </div>
        </div>
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Bullish Predictions</div>
          <div className="text-2xl font-bold text-green-500">
            {worklist.filter(w => w.prediction === 'UP').length}
          </div>
        </div>
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Bearish Predictions</div>
          <div className="text-2xl font-bold text-red-500">
            {worklist.filter(w => w.prediction === 'DOWN').length}
          </div>
        </div>
      </div>

      {/* Worklist Table */}
      {worklist.length === 0 ? (
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-12 text-center">
          <div className="text-6xl mb-4">=�</div>
          <h3 className="text-lg font-semibold text-ibkr-text mb-2">No Symbols in Worklist</h3>
          <p className="text-sm text-ibkr-text-secondary mb-4">
            Add symbols manually or use the IBKR Scanner to populate your worklist
          </p>
          <div className="flex items-center justify-center space-x-3">
            <button
              onClick={() => setShowAddDialog(true)}
              className="px-4 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors"
            >
              Add Symbol
            </button>
            <button
              onClick={() => setShowScannerDialog(true)}
              className="px-4 py-2 bg-ibkr-surface border border-ibkr-border hover:bg-ibkr-bg text-ibkr-text rounded transition-colors"
            >
              Use Scanner
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-ibkr-border bg-ibkr-bg">
                  <th className="text-left p-3 text-xs font-semibold text-ibkr-text-secondary">Symbol</th>
                  <th className="text-right p-3 text-xs font-semibold text-ibkr-text-secondary">Price</th>
                  <th className="text-right p-3 text-xs font-semibold text-ibkr-text-secondary">Change</th>
                  <th className="text-right p-3 text-xs font-semibold text-ibkr-text-secondary">Volume</th>
                  <th className="text-center p-3 text-xs font-semibold text-ibkr-text-secondary">Prediction</th>
                  <th className="text-right p-3 text-xs font-semibold text-ibkr-text-secondary">Confidence</th>
                  <th className="text-left p-3 text-xs font-semibold text-ibkr-text-secondary">Analysis</th>
                  <th className="text-left p-3 text-xs font-semibold text-ibkr-text-secondary">Notes</th>
                  <th className="text-center p-3 text-xs font-semibold text-ibkr-text-secondary">Actions</th>
                </tr>
              </thead>
              <tbody>
                {worklist.map((item) => (
                  <tr key={item.symbol} className="border-b border-ibkr-border hover:bg-ibkr-bg transition-colors">
                    <td className="p-3">
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-ibkr-text">{item.symbol}</span>
                        {!item.has_live_data && (
                          <span className="text-xs text-yellow-500" title="No live data">�</span>
                        )}
                      </div>
                    </td>
                    <td className="p-3 text-right text-ibkr-text font-mono">
                      {item.current_price ? `$${item.current_price.toFixed(2)}` : '-'}
                    </td>
                    <td className="p-3 text-right">
                      {item.change_percent !== undefined ? (
                        <span className={item.change_percent >= 0 ? 'text-green-500' : 'text-red-500'}>
                          {item.change_percent >= 0 ? '+' : ''}
                          {item.change_percent.toFixed(2)}%
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="p-3 text-right text-ibkr-text-secondary text-sm">
                      {item.volume ? item.volume.toLocaleString() : '-'}
                    </td>
                    <td className="p-3 text-center">
                      {item.prediction ? (
                        <span
                          className={`px-2 py-1 rounded text-xs font-semibold ${
                            item.prediction === 'UP'
                              ? 'bg-green-500/10 text-green-500 border border-green-500/30'
                              : 'bg-red-500/10 text-red-500 border border-red-500/30'
                          }`}
                        >
                          {item.prediction}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="p-3 text-right text-ibkr-text">
                      {item.confidence ? `${(item.confidence * 100).toFixed(0)}%` : '-'}
                    </td>
                    <td className="p-3 text-sm text-ibkr-text-secondary max-w-xs truncate">
                      {item.analysis || '-'}
                    </td>
                    <td className="p-3 text-sm">
                      {editingNotes === item.symbol ? (
                        <div className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={notesText}
                            onChange={(e) => setNotesText(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') updateNotes(item.symbol, notesText);
                              if (e.key === 'Escape') setEditingNotes(null);
                            }}
                            className="flex-1 px-2 py-1 bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text text-sm focus:outline-none focus:ring-1 focus:ring-ibkr-accent"
                            autoFocus
                          />
                          <button
                            onClick={() => updateNotes(item.symbol, notesText)}
                            className="text-green-500 hover:text-green-400"
                          >
                            
                          </button>
                          <button
                            onClick={() => setEditingNotes(null)}
                            className="text-red-500 hover:text-red-400"
                          >
                            
                          </button>
                        </div>
                      ) : (
                        <div
                          onClick={() => {
                            setEditingNotes(item.symbol);
                            setNotesText(item.notes || '');
                          }}
                          className="cursor-pointer text-ibkr-text-secondary hover:text-ibkr-text"
                        >
                          {item.notes || 'Click to add notes...'}
                        </div>
                      )}
                    </td>
                    <td className="p-3 text-center">
                      <button
                        onClick={() => removeSymbol(item.symbol)}
                        className="text-red-500 hover:text-red-400 transition-colors"
                        title="Remove from worklist"
                      >
                        =�
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Add Symbol Dialog */}
      {showAddDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-bold text-ibkr-text mb-4">Add Symbol to Worklist</h3>
            <input
              type="text"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === 'Enter' && addSymbol()}
              placeholder="Enter symbol (e.g., AAPL)"
              className="w-full px-4 py-2 bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:ring-2 focus:ring-ibkr-accent mb-4"
              autoFocus
            />
            <div className="flex items-center space-x-3">
              <button
                onClick={addSymbol}
                disabled={!newSymbol.trim()}
                className="flex-1 px-4 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors disabled:opacity-50"
              >
                Add Symbol
              </button>
              <button
                onClick={() => {
                  setShowAddDialog(false);
                  setNewSymbol('');
                }}
                className="flex-1 px-4 py-2 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text rounded transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Scanner Dialog */}
      {showScannerDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-ibkr-surface border border-ibkr-border rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-ibkr-border">
              <h3 className="text-lg font-bold text-ibkr-text mb-4">IBKR Market Scanner</h3>
              <div className="flex items-center space-x-3">
                <select
                  value={selectedScanner}
                  onChange={(e) => setSelectedScanner(e.target.value)}
                  className="flex-1 px-4 py-2 bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:ring-2 focus:ring-ibkr-accent"
                >
                  {scannerPresets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name} - {preset.description}
                    </option>
                  ))}
                </select>
                <button
                  onClick={runScanner}
                  disabled={scanning}
                  className="px-6 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors disabled:opacity-50"
                >
                  {scanning ? 'Scanning...' : 'Run Scanner'}
                </button>
              </div>
            </div>

            {scannerResults.length > 0 && (
              <>
                <div className="flex-1 overflow-y-auto p-6">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-ibkr-surface">
                      <tr className="border-b border-ibkr-border">
                        <th className="text-left p-2 text-xs font-semibold text-ibkr-text-secondary">
                          <input
                            type="checkbox"
                            onChange={(e) => {
                              setScannerResults(prev =>
                                prev.map(r => ({ ...r, selected: e.target.checked }))
                              );
                            }}
                            className="rounded"
                          />
                        </th>
                        <th className="text-left p-2 text-xs font-semibold text-ibkr-text-secondary">Rank</th>
                        <th className="text-left p-2 text-xs font-semibold text-ibkr-text-secondary">Symbol</th>
                        <th className="text-right p-2 text-xs font-semibold text-ibkr-text-secondary">Price</th>
                        <th className="text-right p-2 text-xs font-semibold text-ibkr-text-secondary">Change</th>
                        <th className="text-right p-2 text-xs font-semibold text-ibkr-text-secondary">Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scannerResults.map((result) => (
                        <tr
                          key={result.symbol}
                          className="border-b border-ibkr-border hover:bg-ibkr-bg cursor-pointer"
                          onClick={() => toggleScannerSelection(result.symbol)}
                        >
                          <td className="p-2">
                            <input
                              type="checkbox"
                              checked={result.selected || false}
                              onChange={() => toggleScannerSelection(result.symbol)}
                              className="rounded"
                            />
                          </td>
                          <td className="p-2 text-ibkr-text-secondary text-sm">{result.rank}</td>
                          <td className="p-2 text-ibkr-text font-semibold">{result.symbol}</td>
                          <td className="p-2 text-right text-ibkr-text font-mono text-sm">
                            ${result.price?.toFixed(2) || '-'}
                          </td>
                          <td className="p-2 text-right">
                            {result.change_percent !== undefined ? (
                              <span className={result.change_percent >= 0 ? 'text-green-500' : 'text-red-500'}>
                                {result.change_percent >= 0 ? '+' : ''}
                                {result.change_percent.toFixed(2)}%
                              </span>
                            ) : (
                              '-'
                            )}
                          </td>
                          <td className="p-2 text-right text-ibkr-text-secondary text-sm">
                            {result.volume?.toLocaleString() || '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="p-6 border-t border-ibkr-border flex items-center justify-between">
                  <div className="text-sm text-ibkr-text-secondary">
                    {scannerResults.filter(r => r.selected).length} of {scannerResults.length} selected
                  </div>
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={() => {
                        setShowScannerDialog(false);
                        setScannerResults([]);
                      }}
                      className="px-4 py-2 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text rounded transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={addSelectedToWorklist}
                      disabled={scannerResults.filter(r => r.selected).length === 0}
                      className="px-6 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors disabled:opacity-50"
                    >
                      Add Selected to Worklist
                    </button>
                  </div>
                </div>
              </>
            )}

            {!scanning && scannerResults.length === 0 && (
              <div className="flex-1 flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="text-6xl mb-4">=</div>
                  <p className="text-ibkr-text-secondary">
                    Select a scanner preset and click "Run Scanner" to find stocks
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Worklist;
