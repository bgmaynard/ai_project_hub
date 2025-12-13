"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FOP = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * A Future Option Contract
 */
class FOP {
    constructor(symbol, expiry, strike, right, multiplier, exchange, currency) {
        this.symbol = symbol;
        this.expiry = expiry;
        this.strike = strike;
        this.right = right;
        this.multiplier = multiplier;
        this.exchange = exchange;
        this.currency = currency;
        this.secType = sec_type_1.default.FOP;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "GLOBEX";
        this.multiplier = this.multiplier ?? 50;
    }
}
exports.FOP = FOP;
exports.default = FOP;
//# sourceMappingURL=fop.js.map