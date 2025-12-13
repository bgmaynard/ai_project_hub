"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Bond = void 0;
const sec_type_1 = __importDefault(require("../data/enum/sec-type"));
/**
 * A Bond Contract
 */
class Bond {
    constructor(symbol, maturity, exchange, currency) {
        this.symbol = symbol;
        this.maturity = maturity;
        this.exchange = exchange;
        this.currency = currency;
        this.secType = sec_type_1.default.BOND;
        this.currency = this.currency ?? "USD";
    }
    get lastTradeDateOrContractMonth() {
        return this.maturity;
    }
}
exports.Bond = Bond;
exports.default = Bond;
//# sourceMappingURL=bond.js.map