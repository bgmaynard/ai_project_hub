/**
 * WhatToShow.
 * @see https://interactivebrokers.github.io/tws-api/historical_bars.html#hd_what_to_show
 */
export declare const WhatToShow: {
    readonly None: "";
    readonly TRADES: "TRADES";
    readonly MIDPOINT: "MIDPOINT";
    readonly BID: "BID";
    readonly ASK: "ASK";
    readonly BID_ASK: "BID_ASK";
    readonly HISTORICAL_VOLATILITY: "HISTORICAL_VOLATILITY";
    readonly OPTION_IMPLIED_VOLATILITY: "OPTION_IMPLIED_VOLATILITY";
    readonly YIELD_ASK: "YIELD_ASK";
    readonly YIELD_BID: "YIELD_BID";
    readonly YIELD_BID_ASK: "YIELD_BID_ASK";
    readonly YIELD_LAST: "YIELD_LAST";
    readonly ADJUSTED_LAST: "ADJUSTED_LAST";
    readonly SCHEDULE: "SCHEDULE";
    readonly AGGTRADES: "AGGTRADES";
};
export type WhatToShow = (typeof WhatToShow)[keyof typeof WhatToShow];
