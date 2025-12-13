"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MarketDataType = void 0;
/**
 * TWS market data types.
 */
var MarketDataType;
(function (MarketDataType) {
    /* Disables frozen, delayed and delayed-frozen market data. */
    MarketDataType[MarketDataType["REALTIME"] = 1] = "REALTIME";
    /* Enables frozen market data. */
    MarketDataType[MarketDataType["FROZEN"] = 2] = "FROZEN";
    /* Enables delayed and disables delayed-frozen market data. */
    MarketDataType[MarketDataType["DELAYED"] = 3] = "DELAYED";
    /* Enables delayed and delayed-frozen market data. */
    MarketDataType[MarketDataType["DELAYED_FROZEN"] = 4] = "DELAYED_FROZEN";
})(MarketDataType || (exports.MarketDataType = MarketDataType = {}));
//# sourceMappingURL=market-data-type.js.map