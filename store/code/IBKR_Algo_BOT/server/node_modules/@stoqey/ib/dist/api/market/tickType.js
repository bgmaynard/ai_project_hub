"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TickType = void 0;
/**
 * Market data tick types.
 *
 * @see https://interactivebrokers.github.io/tws-api/tick_types.html
 */
var TickType;
(function (TickType) {
    /** Number of contracts or lots offered at the bid price. */
    TickType[TickType["BID_SIZE"] = 0] = "BID_SIZE";
    /** Highest priced bid for the contract. */
    TickType[TickType["BID"] = 1] = "BID";
    /** Lowest price offer on the contract.. */
    TickType[TickType["ASK"] = 2] = "ASK";
    /** Number of contracts or lots offered at the ask price. */
    TickType[TickType["ASK_SIZE"] = 3] = "ASK_SIZE";
    /** Last price at which the contract traded. */
    TickType[TickType["LAST"] = 4] = "LAST";
    /** Number of contracts or lots traded at the last price. */
    TickType[TickType["LAST_SIZE"] = 5] = "LAST_SIZE";
    /** High price for the day. */
    TickType[TickType["HIGH"] = 6] = "HIGH";
    /** Low price for the day. */
    TickType[TickType["LOW"] = 7] = "LOW";
    /** Trading volume for the day for the selected contract (US Stocks: multiplier 100). */
    TickType[TickType["VOLUME"] = 8] = "VOLUME";
    /**
     * The last available closing price for the previous day.
     * For US Equities, we use corporate action processing to get the closing price,
     * so the close price is adjusted to reflect forward and reverse splits and cash and stock dividends.
     */
    TickType[TickType["CLOSE"] = 9] = "CLOSE";
    /** Computed Greeks for the underlying stock price and the option reference price. */
    TickType[TickType["BID_OPTION"] = 10] = "BID_OPTION";
    /** Computed Greeks for the underlying stock price and the option reference price. */
    TickType[TickType["ASK_OPTION"] = 11] = "ASK_OPTION";
    /** Computed Greeks for the underlying stock price and the option reference price */
    TickType[TickType["LAST_OPTION"] = 12] = "LAST_OPTION";
    /** Computed Greeks and model's implied volatility for the underlying stock price and the option reference price. */
    TickType[TickType["MODEL_OPTION"] = 13] = "MODEL_OPTION";
    /** Today's opening price. */
    TickType[TickType["OPEN"] = 14] = "OPEN";
    /** Lowest price for the last 13 weeks. */
    TickType[TickType["LOW_13_WEEK"] = 15] = "LOW_13_WEEK";
    /** Highest price for the last 13 weeks. */
    TickType[TickType["HIGH_13_WEEK"] = 16] = "HIGH_13_WEEK";
    /** Lowest price for the last 26 weeks. */
    TickType[TickType["LOW_26_WEEK"] = 17] = "LOW_26_WEEK";
    /** Highest price for the last 26 weeks. */
    TickType[TickType["HIGH_26_WEEK"] = 18] = "HIGH_26_WEEK";
    /** Lowest price for the last 52 weeks. */
    TickType[TickType["LOW_52_WEEK"] = 19] = "LOW_52_WEEK";
    /** Highest price for the last 52 weeks. */
    TickType[TickType["HIGH_52_WEEK"] = 20] = "HIGH_52_WEEK";
    /** The average daily trading volume over 90 days (multiply this value times 100). */
    TickType[TickType["AVG_VOLUME"] = 21] = "AVG_VOLUME";
    /** Total number of options that were not closed. */
    TickType[TickType["OPEN_INTEREST"] = 22] = "OPEN_INTEREST";
    /** The 30-day historical volatility (currently for stocks). */
    TickType[TickType["OPTION_HISTORICAL_VOL"] = 23] = "OPTION_HISTORICAL_VOL";
    /**
     * A prediction of how volatile an underlying will be in the future.
     * The IB 30-day volatility is the at-market volatility estimated for a maturity thirty calendar days forward of the current trading day,
     * and is based on option prices from two consecutive expiration months.
     */
    TickType[TickType["OPTION_IMPLIED_VOL"] = 24] = "OPTION_IMPLIED_VOL";
    /** Not Used. */
    TickType[TickType["OPTION_BID_EXCH"] = 25] = "OPTION_BID_EXCH";
    /** Not Used. */
    TickType[TickType["OPTION_ASK_EXCH"] = 26] = "OPTION_ASK_EXCH";
    /**	Call option open interest. */
    TickType[TickType["OPTION_CALL_OPEN_INTEREST"] = 27] = "OPTION_CALL_OPEN_INTEREST";
    /** Put option open interest. */
    TickType[TickType["OPTION_PUT_OPEN_INTEREST"] = 28] = "OPTION_PUT_OPEN_INTEREST";
    /** Call option volume for the trading day. */
    TickType[TickType["OPTION_CALL_VOLUME"] = 29] = "OPTION_CALL_VOLUME";
    /** Put option volume for the trading day. */
    TickType[TickType["OPTION_PUT_VOLUME"] = 30] = "OPTION_PUT_VOLUME";
    /** The number of points that the index is over the cash index. */
    TickType[TickType["INDEX_FUTURE_PREMIUM"] = 31] = "INDEX_FUTURE_PREMIUM";
    /** Identifies the options exchange(s) posting the best bid price on the options contract. */
    TickType[TickType["BID_EXCH"] = 32] = "BID_EXCH";
    /** Identifies the options exchange(s) posting the best ask price on the options contract. */
    TickType[TickType["ASK_EXCH"] = 33] = "ASK_EXCH";
    /** The number of shares that would trade if no new orders were received and the auction were held now. */
    TickType[TickType["AUCTION_VOLUME"] = 34] = "AUCTION_VOLUME";
    /**
     * The price at which the auction would occur if no new orders were received and the auction were held now.
     * The indicative price for the auction.
     */
    TickType[TickType["AUCTION_PRICE"] = 35] = "AUCTION_PRICE";
    /**
     * The number of unmatched shares for the next auction; returns how many more shares are on one side of the auction than the other.
     */
    TickType[TickType["AUCTION_IMBALANCE"] = 36] = "AUCTION_IMBALANCE";
    /**
     * The mark price is equal to the Last Price unless: Ask < Last - the mark price is equal to the Ask Price.
     * Bid > Last - the mark price is equal to the Bid Price.
     */
    TickType[TickType["MARK_PRICE"] = 37] = "MARK_PRICE";
    /** Computed EFP bid price. */
    TickType[TickType["BID_EFP_COMPUTATION"] = 38] = "BID_EFP_COMPUTATION";
    /** Computed EFP ask price. */
    TickType[TickType["ASK_EFP_COMPUTATION"] = 39] = "ASK_EFP_COMPUTATION";
    /** Computed EFP last price. */
    TickType[TickType["LAST_EFP_COMPUTATION"] = 40] = "LAST_EFP_COMPUTATION";
    /** Computed EFP open price. */
    TickType[TickType["OPEN_EFP_COMPUTATION"] = 41] = "OPEN_EFP_COMPUTATION";
    /** Computed high EFP traded price for the day. */
    TickType[TickType["HIGH_EFP_COMPUTATION"] = 42] = "HIGH_EFP_COMPUTATION";
    /** Computed low EFP traded price for the day. */
    TickType[TickType["LOW_EFP_COMPUTATION"] = 43] = "LOW_EFP_COMPUTATION";
    /** Computed closing EFP traded price for the day. */
    TickType[TickType["CLOSE_EFP_COMPUTATION"] = 44] = "CLOSE_EFP_COMPUTATION";
    /** Time of the last trade (in UNIX time). */
    TickType[TickType["LAST_TIMESTAMP"] = 45] = "LAST_TIMESTAMP";
    /** Describes the level of difficulty with which the contract can be sold short. */
    TickType[TickType["SHORTABLE"] = 46] = "SHORTABLE";
    /** Provides the available Reuter's Fundamental Ratios. */
    TickType[TickType["FUNDAMENTAL_RATIOS"] = 47] = "FUNDAMENTAL_RATIOS";
    /** Last trade details. */
    TickType[TickType["RT_VOLUME"] = 48] = "RT_VOLUME";
    /** Indicates if a contract is halted */
    TickType[TickType["HALTED"] = 49] = "HALTED";
    /** Implied yield of the bond if it is purchased at the current bid. */
    TickType[TickType["BID_YIELD"] = 50] = "BID_YIELD";
    /**	Implied yield of the bond if it is purchased at the current ask. */
    TickType[TickType["ASK_YIELD"] = 51] = "ASK_YIELD";
    /** Implied yield of the bond if it is purchased at the last price. */
    TickType[TickType["LAST_YIELD"] = 52] = "LAST_YIELD";
    /** Greek values are based off a user customized price. */
    TickType[TickType["CUST_OPTION_COMPUTATION"] = 53] = "CUST_OPTION_COMPUTATION";
    /** Trade count for the day. */
    TickType[TickType["TRADE_COUNT"] = 54] = "TRADE_COUNT";
    /** Trade count per minute. */
    TickType[TickType["TRADE_RATE"] = 55] = "TRADE_RATE";
    /** Volume per minute. */
    TickType[TickType["VOLUME_RATE"] = 56] = "VOLUME_RATE";
    /** Last Regular Trading Hours traded price. */
    TickType[TickType["LAST_RTH_TRADE"] = 57] = "LAST_RTH_TRADE";
    /** 30-day real time historical volatility. */
    TickType[TickType["RT_HISTORICAL_VOL"] = 58] = "RT_HISTORICAL_VOL";
    /** Contract's dividends. */
    TickType[TickType["IB_DIVIDENDS"] = 59] = "IB_DIVIDENDS";
    /** The bond factor is a number that indicates the ratio of the current bond principal to the original principal. */
    TickType[TickType["BOND_FACTOR_MULTIPLIER"] = 60] = "BOND_FACTOR_MULTIPLIER";
    /**
     * The imbalance that is used to determine which at-the-open or at-the-close orders can be
     * entered following the publishing of the regulatory imbalance.
     */
    TickType[TickType["REGULATORY_IMBALANCE"] = 61] = "REGULATORY_IMBALANCE";
    /** Contract's news feed. */
    TickType[TickType["NEWS_TICK"] = 62] = "NEWS_TICK";
    /** The past three minutes volume. Interpolation may be applied. */
    TickType[TickType["SHORT_TERM_VOLUME_3_MIN"] = 63] = "SHORT_TERM_VOLUME_3_MIN";
    /** The past five minutes volume. Interpolation may be applied. */
    TickType[TickType["SHORT_TERM_VOLUME_5_MIN"] = 64] = "SHORT_TERM_VOLUME_5_MIN";
    /** The past ten minutes volume. Interpolation may be applied. */
    TickType[TickType["SHORT_TERM_VOLUME_10_MIN"] = 65] = "SHORT_TERM_VOLUME_10_MIN";
    /** Delayed bid price */
    TickType[TickType["DELAYED_BID"] = 66] = "DELAYED_BID";
    /** Delayed ask price. */
    TickType[TickType["DELAYED_ASK"] = 67] = "DELAYED_ASK";
    /** Delayed last traded price.  */
    TickType[TickType["DELAYED_LAST"] = 68] = "DELAYED_LAST";
    /** Delayed bid size. */
    TickType[TickType["DELAYED_BID_SIZE"] = 69] = "DELAYED_BID_SIZE";
    /** Delayed ask size. */
    TickType[TickType["DELAYED_ASK_SIZE"] = 70] = "DELAYED_ASK_SIZE";
    /** Delayed last size. */
    TickType[TickType["DELAYED_LAST_SIZE"] = 71] = "DELAYED_LAST_SIZE";
    /** Delayed highest price of the day.  */
    TickType[TickType["DELAYED_HIGH"] = 72] = "DELAYED_HIGH";
    /** Delayed lowest price of the day.  */
    TickType[TickType["DELAYED_LOW"] = 73] = "DELAYED_LOW";
    /** Delayed traded volume of the day.  */
    TickType[TickType["DELAYED_VOLUME"] = 74] = "DELAYED_VOLUME";
    /** Delayed close price of the day.  */
    TickType[TickType["DELAYED_CLOSE"] = 75] = "DELAYED_CLOSE";
    /** Delayed open price of the day.  */
    TickType[TickType["DELAYED_OPEN"] = 76] = "DELAYED_OPEN";
    /** Last trade details that excludes "Unreportable Trades" */
    TickType[TickType["RT_TRD_VOLUME"] = 77] = "RT_TRD_VOLUME";
    /* Not currently available. */
    TickType[TickType["CREDITMAN_MARK_PRICE"] = 78] = "CREDITMAN_MARK_PRICE";
    /** Slower mark price update used in system calculations */
    TickType[TickType["CREDITMAN_SLOW_MARK_PRICE"] = 79] = "CREDITMAN_SLOW_MARK_PRICE";
    /** Delayed computed Greeks for the underlying stock price and the option reference price.. */
    TickType[TickType["DELAYED_BID_OPTION"] = 80] = "DELAYED_BID_OPTION";
    /** Delayed computed Greeks for the underlying stock price and the option reference price. */
    TickType[TickType["DELAYED_ASK_OPTION"] = 81] = "DELAYED_ASK_OPTION";
    /** Delayed computed Greeks for the underlying stock price and the option reference price. */
    TickType[TickType["DELAYED_LAST_OPTION"] = 82] = "DELAYED_LAST_OPTION";
    /** Delayed computed Greeks and model's implied volatility for the underlying stock price and the option reference price. */
    TickType[TickType["DELAYED_MODEL_OPTION"] = 83] = "DELAYED_MODEL_OPTION";
    /** Exchange of last traded price. */
    TickType[TickType["LAST_EXCH"] = 84] = "LAST_EXCH";
    /** Timestamp (in Unix ms time) of last trade returned with regulatory snapshot. */
    TickType[TickType["LAST_REG_TIME"] = 85] = "LAST_REG_TIME";
    /** Total number of outstanding futures contracts (TWS v965+). *HSI open interest requested with generic tick 101. */
    TickType[TickType["FUTURES_OPEN_INTEREST"] = 86] = "FUTURES_OPEN_INTEREST";
    /** Average volume of the corresponding option contracts(TWS Build 970+ is required). */
    TickType[TickType["AVG_OPT_VOLUME"] = 87] = "AVG_OPT_VOLUME";
    /** Delayed time of the last trade (in UNIX time) (TWS Build 970+ is required) */
    TickType[TickType["DELAYED_LAST_TIMESTAMP"] = 88] = "DELAYED_LAST_TIMESTAMP";
    /** Number of shares available to short (TWS Build 974+ is required) */
    TickType[TickType["SHORTABLE_SHARES"] = 89] = "SHORTABLE_SHARES";
    TickType[TickType["DELAYED_HALTED"] = 90] = "DELAYED_HALTED";
    TickType[TickType["REUTERS_2_MUTUAL_FUNDS"] = 91] = "REUTERS_2_MUTUAL_FUNDS";
    /**
     * Today's closing price of ETF's Net Asset Value (NAV).
     * Calculation is based on prices of ETF's underlying securities.
     */
    TickType[TickType["ETF_NAV_CLOSE"] = 92] = "ETF_NAV_CLOSE";
    /**
     * Yesterday's closing price of ETF's Net Asset Value (NAV).
     * Calculation is based on prices of ETF's underlying securities.
     */
    TickType[TickType["ETF_NAV_PRIOR_CLOSE"] = 93] = "ETF_NAV_PRIOR_CLOSE";
    /**
     * The bid price of ETF's Net Asset Value (NAV).
     * Calculation is based on prices of ETF's underlying securities.
     */
    TickType[TickType["ETF_NAV_BID"] = 94] = "ETF_NAV_BID";
    /**
     * The ask price of ETF's Net Asset Value (NAV).
     * Calculation is based on prices of ETF's underlying securities.
     */
    TickType[TickType["ETF_NAV_ASK"] = 95] = "ETF_NAV_ASK";
    /**
     * The last price of Net Asset Value (NAV).
     * For ETFs: Calculation is based on prices of ETF's underlying securities.
     * For NextShares: Value is provided by NASDAQ.
     */
    TickType[TickType["ETF_NAV_LAST"] = 96] = "ETF_NAV_LAST";
    /** ETF Nav Last for Frozen data. */
    TickType[TickType["ETF_NAV_FROZEN_LAST"] = 97] = "ETF_NAV_FROZEN_LAST";
    /** The high price of ETF's Net Asset Value (NAV). */
    TickType[TickType["ETF_NAV_HIGH"] = 98] = "ETF_NAV_HIGH";
    /** The low price of ETF's Net Asset Value (NAV). */
    TickType[TickType["ETF_NAV_LOW"] = 99] = "ETF_NAV_LOW";
    /** TBD */
    TickType[TickType["SOCIAL_MARKET_ANALYTICS"] = 100] = "SOCIAL_MARKET_ANALYTICS";
    /** Midpoint is calculated based on IPO price range */
    TickType[TickType["ESTIMATED_IPO_MIDPOINT"] = 101] = "ESTIMATED_IPO_MIDPOINT";
    /** Final price for IPO */
    TickType[TickType["FINAL_IPO_LAST"] = 102] = "FINAL_IPO_LAST";
    /** TBD */
    TickType[TickType["DELAYED_YIELD_BID"] = 103] = "DELAYED_YIELD_BID";
    /** TBD */
    TickType[TickType["DELAYED_YIELD_ASK"] = 104] = "DELAYED_YIELD_ASK";
    TickType[TickType["UNKNOWN"] = 2147483647] = "UNKNOWN";
})(TickType || (exports.TickType = TickType = {}));
exports.default = TickType;
//# sourceMappingURL=tickType.js.map