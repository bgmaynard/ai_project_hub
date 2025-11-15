// Training Configuration
export interface TrainingConfig {
  model_type: string;
  symbols: string[];
  timeframes: string[];
  features: string[];
  start_date: string;
  end_date: string;
  train_split: number;
}

// Training Metrics
export interface TrainingMetrics {
  epoch: number;
  total_epochs: number;
  train_accuracy: number;
  val_accuracy: number;
  train_loss: number;
  val_loss: number;
  status: 'training' | 'complete' | 'error';
  time_elapsed?: number;
  eta?: number;
}

// Model Performance
export interface ModelPerformance {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  status: 'active' | 'standby' | 'error';
  change?: number;
}

// Backtest Configuration
export interface BacktestConfig {
  strategy: string;
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_capital: number;
  entry_rules: EntryRules;
  exit_rules: ExitRules;
  risk_management: RiskManagement;
}

export interface EntryRules {
  gap_percentage: number;
  volume_requirement: string;
  ai_confidence_threshold: number;
  news_catalyst: boolean;
  custom_rules?: string[];
}

export interface ExitRules {
  target_profit_pct: number;
  stop_loss_pct: number;
  time_based_stop?: string;
  trailing_stop_pct?: number;
}

export interface RiskManagement {
  max_position_size_pct: number;
  max_daily_trades: number;
  daily_loss_limit_pct: number;
  max_drawdown_pct: number;
}

// Backtest Results
export interface BacktestResults {
  backtest_id: string;
  performance: PerformanceMetrics;
  equity_curve: EquityPoint[];
  trades: Trade[];
  monthly_returns: MonthlyReturn[];
  drawdown_data: DrawdownPoint[];
}

export interface BacktestResponse {
  backtest_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  results?: BacktestResults;
  error?: string;
}

export interface PerformanceMetrics {
  total_return: number;
  cagr: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  expectancy: number;
}

export interface EquityPoint {
  timestamp: string;
  value: number;
}

export interface Trade {
  trade_id: string;
  symbol: string;
  date: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_pct: number;
  hold_time: number;
  winner: boolean;
  setup_conditions: SetupConditions;
}

export interface SetupConditions {
  gap_pct: number;
  volume_ratio: number;
  ai_confidence: number;
  news_catalyst?: string;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
}

export interface DrawdownPoint {
  timestamp: string;
  drawdown_pct: number;
}

// Live Predictions
export interface Prediction {
  symbol: string;
  price: number;
  signal: 'BUY' | 'SELL';
  confidence: number;
  strategy: string;
  ai_comment?: string;
  timestamp: string;
}

// Alert Configuration
export interface AlertConfig {
  alert_id?: string;
  name: string;
  symbols: string[];
  confidence_threshold: number;
  signal_type: 'BUY' | 'SELL' | 'ANY';
  strategy?: string;
  price_above?: number;
  price_below?: number;
  volume_above?: number;
  notification_methods: NotificationMethods;
}

export interface NotificationMethods {
  desktop: boolean;
  browser_push: boolean;
  email?: string;
  sms?: string;
  sound: boolean;
  sound_file?: string;
}

export interface AlertHistory {
  alert_id: string;
  alert_name: string;
  symbol: string;
  signal: 'BUY' | 'SELL';
  confidence: number;
  price: number;
  timestamp: string;
  action_taken: string;
  notification_sent: boolean;
}

export interface ActiveAlert extends AlertConfig {
  alert_id: string;
  enabled: boolean;
  trigger_count: number;
  last_triggered?: string;
  created_at: string;
}

// TradingView
export interface TradingViewPush {
  symbol: string;
  push_all: boolean;
  elements: TradingViewElements;
  config: TradingViewConfig;
}

export interface TradingViewElements {
  trade_markers: boolean;
  ai_predictions: boolean;
  support_resistance: boolean;
  strategy_signals: boolean;
  risk_zones: boolean;
  volume_profile: boolean;
}

export interface TradingViewConfig {
  color_scheme: ColorScheme;
  size_style: SizeStyle;
  label_visibility: boolean;
}

export interface ColorScheme {
  buy_color: string;
  sell_color: string;
  support_color: string;
  resistance_color: string;
  stop_loss_color: string;
  target_color: string;
}

export interface SizeStyle {
  marker_size: 'small' | 'medium' | 'large';
  line_width: number;
  label_size: 'small' | 'medium' | 'large';
}

// Claude Analysis
export interface ClaudeInsights {
  insights: string;
  concerns: string[];
  suggestions: string[];
  assessment?: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  strengths?: string[];
  weaknesses?: string[];
  recommendations?: string[];
}

// API Response
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// WebSocket Message
export interface WebSocketMessage<T> {
  type: string;
  channel?: string;
  data: T;
  timestamp: string;
}
