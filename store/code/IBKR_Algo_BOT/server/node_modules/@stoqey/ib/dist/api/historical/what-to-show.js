"use strict";
/**
 * WhatToShow.
 * @see https://interactivebrokers.github.io/tws-api/historical_bars.html#hd_what_to_show
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.WhatToShow = void 0;
exports.WhatToShow = {
    None: "",
    // only these are valid for real-time bars
    TRADES: "TRADES",
    MIDPOINT: "MIDPOINT",
    BID: "BID",
    ASK: "ASK",
    BID_ASK: "BID_ASK",
    HISTORICAL_VOLATILITY: "HISTORICAL_VOLATILITY",
    OPTION_IMPLIED_VOLATILITY: "OPTION_IMPLIED_VOLATILITY",
    YIELD_ASK: "YIELD_ASK",
    YIELD_BID: "YIELD_BID",
    YIELD_BID_ASK: "YIELD_BID_ASK",
    YIELD_LAST: "YIELD_LAST",
    ADJUSTED_LAST: "ADJUSTED_LAST",
    SCHEDULE: "SCHEDULE",
    AGGTRADES: "AGGTRADES",
};
//# sourceMappingURL=what-to-show.js.map