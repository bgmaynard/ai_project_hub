"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.aapl_contract = exports.sample_option = exports.sample_future = exports.sample_crypto = exports.sample_dax_index = exports.sample_index = exports.sample_bond = exports.sample_etf = exports.sample_stock = void 0;
/**
 * This file describe sample contracts to be used in various tests code.
 */
const __1 = require("../../..");
const crypto_1 = __importDefault(require("../../../api/contract/crypto"));
exports.sample_stock = new __1.Stock("AAPL");
exports.sample_etf = new __1.Stock("SPY");
exports.sample_bond = new __1.Bond("US064159KJ44");
exports.sample_index = new __1.Index("ES", "USD");
exports.sample_dax_index = new __1.Index("DAX", "EUR", "EUREX");
exports.sample_crypto = new crypto_1.default("BTC");
// This one will need to be updated sometimes
exports.sample_future = new __1.Future("ES", "ESH5", "202503", "CME", 50);
// This one may need to be updated from times to times
exports.sample_option = new __1.Option("SPY", "20260116", 440, __1.OptionType.Call);
/*
   Contracts with conId for tests needing IB's conID
*/
exports.aapl_contract = {
    conId: 265598,
    secType: __1.SecType.STK,
    symbol: "AAPL",
    exchange: "SMART",
    currency: "USD",
};
//# sourceMappingURL=contracts.js.map