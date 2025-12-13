"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CFD = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * A CFD contract.
 */
class CFD {
    constructor(symbol, currency, exchange) {
        this.symbol = symbol;
        this.currency = currency;
        this.exchange = exchange;
        this.secType = sec_type_1.default.STK;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "SMART";
    }
}
exports.CFD = CFD;
exports.default = CFD;
//# sourceMappingURL=cfd.js.map