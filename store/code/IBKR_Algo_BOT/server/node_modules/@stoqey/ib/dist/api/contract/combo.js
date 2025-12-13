"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Combo = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * A Combo contract.
 */
class Combo {
    constructor(symbol, comboLegs, currency, exchange) {
        this.symbol = symbol;
        this.comboLegs = comboLegs;
        this.currency = currency;
        this.exchange = exchange;
        this.secType = sec_type_1.default.BAG;
        this.currency = this.currency ?? "USD";
        this.exchange = this.exchange ?? "SMART";
    }
}
exports.Combo = Combo;
exports.default = Combo;
//# sourceMappingURL=combo.js.map