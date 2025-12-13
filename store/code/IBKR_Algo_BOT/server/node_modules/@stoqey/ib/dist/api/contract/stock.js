"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Stock = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * Stock contract.
 */
class Stock {
    constructor(symbol, exchange, currency) {
        this.symbol = symbol;
        this.exchange = exchange;
        this.currency = currency;
        this.secType = sec_type_1.default.STK;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "SMART";
    }
}
exports.Stock = Stock;
exports.default = Stock;
//# sourceMappingURL=stock.js.map