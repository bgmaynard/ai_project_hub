"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Forex = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * A Forex Contract.
 */
class Forex {
    constructor(symbol, currency) {
        // Swap between symbol and currency if the ordering is incorrect.
        this.symbol = symbol;
        this.currency = currency;
        this.exchange = "IDEALPRO";
        this.secType = sec_type_1.default.CASH;
        let temp;
        if (Forex.CURRENCY_SYMBOL_PRIO.indexOf(symbol) >
            Forex.CURRENCY_SYMBOL_PRIO.indexOf(currency)) {
            temp = this.symbol;
            this.symbol = this.currency;
            this.currency = temp;
        }
    }
}
exports.Forex = Forex;
/**
 * Between two currencies,
 * whatever currency comes first should be in "symbol" and the other one must be in "currency".
 */
Forex.CURRENCY_SYMBOL_PRIO = [
    "KRW",
    "EUR",
    "GBP",
    "AUD",
    "USD",
    "TRY",
    "ZAR",
    "CAD",
    "CHF",
    "MXN",
    "HKD",
    "JPY",
    "INR",
    "NOK",
    "SEK",
    "RUB",
];
exports.default = Forex;
//# sourceMappingURL=forex.js.map