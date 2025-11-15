import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface IndicatorConfig {
  name: string;
  description: string;
  strategy_type: 'momentum' | 'trend' | 'volatility' | 'volume' | 'custom';
  timeframe: string;
  ai_assisted: boolean;
  parameters: IndicatorParameter[];
}

interface IndicatorParameter {
  name: string;
  type: 'number' | 'boolean' | 'string';
  default_value: string | number | boolean;
  description: string;
}

interface SavedIndicator {
  indicator_id: string;
  name: string;
  description: string;
  strategy_type: string;
  pine_script: string;
  created_at: string;
  last_used?: string;
}

export const IndicatorBuilder: React.FC = () => {
  // State Management
  const [config, setConfig] = useState<IndicatorConfig>({
    name: '',
    description: '',
    strategy_type: 'momentum',
    timeframe: '1D',
    ai_assisted: true,
    parameters: []
  });

  const [generatedScript, setGeneratedScript] = useState('');
  const [savedIndicators, setSavedIndicators] = useState<SavedIndicator[]>([]);
  const [selectedIndicator, setSelectedIndicator] = useState<SavedIndicator | null>(null);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'builder' | 'saved'>('builder');

  // Load Saved Indicators
  useEffect(() => {
    loadSavedIndicators();
  }, []);

  const loadSavedIndicators = async () => {
    setLoading(true);
    try {
      const response = await apiService.getSavedIndicators();
      if (response.data) {
        setSavedIndicators(response.data);
      }
    } catch (error) {
      console.error('Failed to load saved indicators:', error);
    } finally {
      setLoading(false);
    }
  };

  // Generate PineScript
  const generateIndicator = async () => {
    if (!config.name) {
      alert('Please enter an indicator name');
      return;
    }

    setGenerating(true);
    try {
      const response = await apiService.generateIndicator(config);
      if (response.data?.script) {
        setGeneratedScript(response.data.script);
      }
    } catch (error) {
      console.error('Failed to generate indicator:', error);
    } finally {
      setGenerating(false);
    }
  };

  // Save Indicator
  const saveIndicator = async () => {
    if (!generatedScript) {
      alert('Please generate an indicator first');
      return;
    }

    setSaving(true);
    try {
      const response = await apiService.saveIndicator({
        name: config.name,
        description: config.description,
        strategy_type: config.strategy_type,
        pine_script: generatedScript,
        parameters: config.parameters
      });

      if (response.data?.indicator_id) {
        await loadSavedIndicators();
        // Clear form
        setConfig({
          name: '',
          description: '',
          strategy_type: 'momentum',
          timeframe: '1D',
          ai_assisted: true,
          parameters: []
        });
        setGeneratedScript('');
        setActiveTab('saved');
      }
    } catch (error) {
      console.error('Failed to save indicator:', error);
    } finally {
      setSaving(false);
    }
  };

  // Delete Indicator
  const deleteIndicator = async (indicatorId: string) => {
    if (!window.confirm('Are you sure you want to delete this indicator?')) {
      return;
    }

    try {
      await apiService.deleteIndicator(indicatorId);
      await loadSavedIndicators();
      if (selectedIndicator?.indicator_id === indicatorId) {
        setSelectedIndicator(null);
      }
    } catch (error) {
      console.error('Failed to delete indicator:', error);
    }
  };

  // Copy to Clipboard
  const copyToClipboard = (script: string) => {
    navigator.clipboard.writeText(script);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Add Parameter
  const addParameter = () => {
    setConfig({
      ...config,
      parameters: [
        ...config.parameters,
        {
          name: '',
          type: 'number',
          default_value: 0,
          description: ''
        }
      ]
    });
  };

  // Remove Parameter
  const removeParameter = (index: number) => {
    setConfig({
      ...config,
      parameters: config.parameters.filter((_, i) => i !== index)
    });
  };

  // Update Parameter
  const updateParameter = (index: number, field: keyof IndicatorParameter, value: any) => {
    const updatedParams = [...config.parameters];
    updatedParams[index] = { ...updatedParams[index], [field]: value };
    setConfig({ ...config, parameters: updatedParams });
  };

  return (
    <div className="px-6 space-y-6">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-ibkr-surface border border-ibkr-border rounded-lg p-1">
        <button
          onClick={() => setActiveTab('builder')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
            activeTab === 'builder'
              ? 'bg-ibkr-accent text-white'
              : 'text-ibkr-text hover:bg-ibkr-bg'
          }`}
        >
          🛠️ Indicator Builder
        </button>
        <button
          onClick={() => setActiveTab('saved')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
            activeTab === 'saved'
              ? 'bg-ibkr-accent text-white'
              : 'text-ibkr-text hover:bg-ibkr-bg'
          }`}
        >
          💾 Saved Indicators ({savedIndicators.length})
        </button>
      </div>

      {/* Builder Tab */}
      {activeTab === 'builder' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Configuration Panel */}
          <div className="space-y-6">
            <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
              <h2 className="text-base font-semibold text-ibkr-text mb-4">
                Indicator Configuration
              </h2>

              {/* Indicator Name */}
              <div className="mb-4">
                <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
                  Indicator Name *
                </label>
                <input
                  type="text"
                  value={config.name}
                  onChange={(e) => setConfig({ ...config, name: e.target.value })}
                  placeholder="e.g., AI Enhanced RSI"
                  className="w-full bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-sm text-ibkr-text placeholder-gray-500"
                />
              </div>

              {/* Description */}
              <div className="mb-4">
                <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
                  Description
                </label>
                <textarea
                  value={config.description}
                  onChange={(e) => setConfig({ ...config, description: e.target.value })}
                  placeholder="Describe what your indicator does..."
                  rows={3}
                  className="w-full bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-sm text-ibkr-text placeholder-gray-500 resize-none"
                />
              </div>

              {/* Strategy Type */}
              <div className="mb-4">
                <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
                  Strategy Type
                </label>
                <select
                  value={config.strategy_type}
                  onChange={(e) => setConfig({ ...config, strategy_type: e.target.value as any })}
                  className="w-full bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-sm text-ibkr-text"
                >
                  <option value="momentum">Momentum</option>
                  <option value="trend">Trend Following</option>
                  <option value="volatility">Volatility</option>
                  <option value="volume">Volume</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              {/* Timeframe */}
              <div className="mb-4">
                <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
                  Default Timeframe
                </label>
                <select
                  value={config.timeframe}
                  onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
                  className="w-full bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-sm text-ibkr-text"
                >
                  <option value="1m">1 Minute</option>
                  <option value="5m">5 Minutes</option>
                  <option value="15m">15 Minutes</option>
                  <option value="1H">1 Hour</option>
                  <option value="4H">4 Hours</option>
                  <option value="1D">1 Day</option>
                  <option value="1W">1 Week</option>
                </select>
              </div>

              {/* AI Assisted */}
              <div className="mb-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.ai_assisted}
                    onChange={(e) => setConfig({ ...config, ai_assisted: e.target.checked })}
                    className="rounded border-ibkr-border text-ibkr-accent focus:ring-ibkr-accent"
                  />
                  <span className="text-xs text-ibkr-text">
                    Use Claude AI to enhance indicator logic
                  </span>
                </label>
              </div>
            </div>

            {/* Parameters */}
            <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-semibold text-ibkr-text">
                  Custom Parameters
                </h2>
                <button
                  onClick={addParameter}
                  className="px-3 py-1.5 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white text-xs rounded transition-colors"
                >
                  + Add Parameter
                </button>
              </div>

              {config.parameters.length === 0 ? (
                <p className="text-xs text-ibkr-text-secondary italic text-center py-4">
                  No parameters added. Click "Add Parameter" to create configurable inputs.
                </p>
              ) : (
                <div className="space-y-3">
                  {config.parameters.map((param, index) => (
                    <div key={index} className="bg-ibkr-bg border border-ibkr-border rounded p-3">
                      <div className="grid grid-cols-2 gap-2 mb-2">
                        <input
                          type="text"
                          value={param.name}
                          onChange={(e) => updateParameter(index, 'name', e.target.value)}
                          placeholder="Parameter name"
                          className="bg-ibkr-surface border border-ibkr-border rounded px-2 py-1 text-xs text-ibkr-text placeholder-gray-500"
                        />
                        <select
                          value={param.type}
                          onChange={(e) => updateParameter(index, 'type', e.target.value)}
                          className="bg-ibkr-surface border border-ibkr-border rounded px-2 py-1 text-xs text-ibkr-text"
                        >
                          <option value="number">Number</option>
                          <option value="boolean">Boolean</option>
                          <option value="string">String</option>
                        </select>
                      </div>
                      <input
                        type={param.type === 'number' ? 'number' : 'text'}
                        value={param.default_value.toString()}
                        onChange={(e) => updateParameter(index, 'default_value',
                          param.type === 'number' ? parseFloat(e.target.value) : e.target.value
                        )}
                        placeholder="Default value"
                        className="w-full bg-ibkr-surface border border-ibkr-border rounded px-2 py-1 text-xs text-ibkr-text placeholder-gray-500 mb-2"
                      />
                      <div className="flex items-center justify-between">
                        <input
                          type="text"
                          value={param.description}
                          onChange={(e) => updateParameter(index, 'description', e.target.value)}
                          placeholder="Description (optional)"
                          className="flex-1 bg-ibkr-surface border border-ibkr-border rounded px-2 py-1 text-xs text-ibkr-text placeholder-gray-500"
                        />
                        <button
                          onClick={() => removeParameter(index)}
                          className="ml-2 text-red-400 hover:text-red-300 text-xs"
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Generate Button */}
            <button
              onClick={generateIndicator}
              disabled={generating || !config.name}
              className="w-full px-6 py-3 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white font-medium rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {generating ? '⟳ Generating PineScript...' : '✨ Generate Indicator'}
            </button>
          </div>

          {/* Preview Panel */}
          <div className="space-y-6">
            <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-semibold text-ibkr-text">
                  Generated PineScript
                </h2>
                {generatedScript && (
                  <div className="flex space-x-2">
                    <button
                      onClick={() => copyToClipboard(generatedScript)}
                      className="px-3 py-1.5 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text text-xs rounded transition-colors"
                    >
                      {copied ? '✓ Copied!' : '📋 Copy'}
                    </button>
                    <button
                      onClick={saveIndicator}
                      disabled={saving}
                      className="px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white text-xs rounded transition-colors disabled:opacity-50"
                    >
                      {saving ? '⟳ Saving...' : '💾 Save'}
                    </button>
                  </div>
                )}
              </div>

              {!generatedScript ? (
                <div className="bg-ibkr-bg border border-ibkr-border rounded p-8 text-center">
                  <div className="text-4xl mb-3">📊</div>
                  <p className="text-sm text-ibkr-text-secondary">
                    Configure your indicator and click "Generate" to create PineScript code
                  </p>
                </div>
              ) : (
                <div className="bg-ibkr-bg border border-ibkr-border rounded overflow-hidden">
                  <pre className="p-4 text-xs text-ibkr-text font-mono overflow-x-auto max-h-96 overflow-y-auto">
                    {generatedScript}
                  </pre>
                </div>
              )}
            </div>

            {generatedScript && (
              <div className="bg-blue-500/10 border border-blue-500/50 rounded-lg p-4">
                <p className="text-xs text-blue-400 font-medium mb-2">💡 How to use:</p>
                <ol className="text-xs text-blue-300 space-y-1 ml-4 list-decimal">
                  <li>Copy the generated PineScript code</li>
                  <li>Open TradingView and go to Pine Editor</li>
                  <li>Paste the code and click "Add to Chart"</li>
                  <li>Customize parameters in indicator settings</li>
                </ol>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Saved Indicators Tab */}
      {activeTab === 'saved' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Indicators List */}
          <div className="lg:col-span-1 bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
            <h2 className="text-base font-semibold text-ibkr-text mb-4">
              Your Indicators
            </h2>

            {loading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ibkr-accent mx-auto"></div>
              </div>
            ) : savedIndicators.length === 0 ? (
              <p className="text-xs text-ibkr-text-secondary text-center py-8">
                No saved indicators yet. Create one using the builder!
              </p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {savedIndicators.map((indicator) => (
                  <div
                    key={indicator.indicator_id}
                    onClick={() => setSelectedIndicator(indicator)}
                    className={`p-3 border rounded cursor-pointer transition-colors ${
                      selectedIndicator?.indicator_id === indicator.indicator_id
                        ? 'bg-ibkr-accent/20 border-ibkr-accent'
                        : 'bg-ibkr-bg border-ibkr-border hover:border-ibkr-accent/50'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-ibkr-text">{indicator.name}</p>
                        <p className="text-xs text-ibkr-text-secondary mt-1 line-clamp-2">
                          {indicator.description || 'No description'}
                        </p>
                        <div className="flex items-center space-x-2 mt-2">
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-ibkr-accent/20 text-ibkr-accent">
                            {indicator.strategy_type}
                          </span>
                          <span className="text-xs text-ibkr-text-secondary">
                            {new Date(indicator.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Indicator Details */}
          <div className="lg:col-span-2 bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
            {!selectedIndicator ? (
              <div className="text-center py-16">
                <div className="text-6xl mb-4">📊</div>
                <p className="text-base font-semibold text-ibkr-text mb-2">
                  No Indicator Selected
                </p>
                <p className="text-sm text-ibkr-text-secondary">
                  Select an indicator from the list to view details
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-xl font-bold text-ibkr-text mb-1">
                      {selectedIndicator.name}
                    </h2>
                    <p className="text-sm text-ibkr-text-secondary">
                      {selectedIndicator.description || 'No description provided'}
                    </p>
                  </div>
                  <button
                    onClick={() => deleteIndicator(selectedIndicator.indicator_id)}
                    className="px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs rounded transition-colors"
                  >
                    🗑️ Delete
                  </button>
                </div>

                {/* Metadata */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-ibkr-bg border border-ibkr-border rounded p-3">
                    <p className="text-xs text-ibkr-text-secondary mb-1">Strategy Type</p>
                    <p className="text-sm font-medium text-ibkr-text capitalize">
                      {selectedIndicator.strategy_type}
                    </p>
                  </div>
                  <div className="bg-ibkr-bg border border-ibkr-border rounded p-3">
                    <p className="text-xs text-ibkr-text-secondary mb-1">Created</p>
                    <p className="text-sm font-medium text-ibkr-text">
                      {new Date(selectedIndicator.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="bg-ibkr-bg border border-ibkr-border rounded p-3">
                    <p className="text-xs text-ibkr-text-secondary mb-1">Last Used</p>
                    <p className="text-sm font-medium text-ibkr-text">
                      {selectedIndicator.last_used
                        ? new Date(selectedIndicator.last_used).toLocaleDateString()
                        : 'Never'
                      }
                    </p>
                  </div>
                </div>

                {/* PineScript Code */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-ibkr-text">
                      PineScript Code
                    </h3>
                    <button
                      onClick={() => copyToClipboard(selectedIndicator.pine_script)}
                      className="px-3 py-1.5 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white text-xs rounded transition-colors"
                    >
                      {copied ? '✓ Copied!' : '📋 Copy Code'}
                    </button>
                  </div>
                  <div className="bg-ibkr-bg border border-ibkr-border rounded overflow-hidden">
                    <pre className="p-4 text-xs text-ibkr-text font-mono overflow-x-auto max-h-96 overflow-y-auto">
                      {selectedIndicator.pine_script}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default IndicatorBuilder;
