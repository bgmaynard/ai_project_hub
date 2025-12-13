"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Option = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * Option contact.
 */
class Option {
    constructor(symbol, expiry, strike, right, exchange, currency) {
        this.symbol = symbol;
        this.expiry = expiry;
        this.strike = strike;
        this.right = right;
        this.exchange = exchange;
        this.currency = currency;
        this.secType = sec_type_1.default.OPT;
        this.multiplier = 100;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "SMART";
    }
    get lastTradeDateOrContractMonth() {
        return this.expiry;
    }
}
exports.Option = Option;
exports.default = Option;
//# sourceMappingURL=option.js.map