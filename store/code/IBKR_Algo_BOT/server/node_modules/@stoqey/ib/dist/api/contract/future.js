"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Future = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * A Future Contract
 */
class Future {
    constructor(symbol, localSymbol, lastTradeDateOrContractMonth, exchange, multiplier, currency) {
        this.symbol = symbol;
        this.localSymbol = localSymbol;
        this.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth;
        this.exchange = exchange;
        this.multiplier = multiplier;
        this.currency = currency;
        this.secType = sec_type_1.default.FUT;
        this.currency = this.currency ?? "USD";
    }
}
exports.Future = Future;
exports.default = Future;
//# sourceMappingURL=future.js.map