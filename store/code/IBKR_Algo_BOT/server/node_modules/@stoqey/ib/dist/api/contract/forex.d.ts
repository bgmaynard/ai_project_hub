import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * A Forex Contract.
 */
export declare class Forex implements Contract {
    symbol: string;
    currency: string;
    /**
     * Between two currencies,
     * whatever currency comes first should be in "symbol" and the other one must be in "currency".
     */
    private static readonly CURRENCY_SYMBOL_PRIO;
    constructor(symbol: string, currency: string);
    exchange: string;
    secType: SecType;
}
export default Forex;
