"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Crypto = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * Crypto contract.
 */
class Crypto {
    constructor(symbol, exchange, currency) {
        this.symbol = symbol;
        this.exchange = exchange;
        this.currency = currency;
        this.secType = sec_type_1.default.CRYPTO;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "PAXOS";
    }
}
exports.Crypto = Crypto;
exports.default = Crypto;
//# sourceMappingURL=crypto.js.map