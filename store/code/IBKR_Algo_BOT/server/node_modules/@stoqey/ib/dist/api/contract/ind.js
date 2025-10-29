"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Index = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * Index contract.
 */
class Index {
    constructor(symbol, currency, exchange) {
        this.symbol = symbol;
        this.currency = currency;
        this.exchange = exchange;
        this.secType = sec_type_1.default.IND;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "CME";
    }
}
exports.Index = Index;
exports.default = Index;
//# sourceMappingURL=ind.js.map