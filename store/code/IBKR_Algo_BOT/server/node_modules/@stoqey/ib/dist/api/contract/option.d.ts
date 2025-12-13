import OptionType from "../data/enum/option-type";
import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * Option contact.
 */
export declare class Option implements Contract {
    symbol: string;
    expiry: string;
    strike: number;
    right: OptionType;
    exchange?: string;
    currency?: string;
    constructor(symbol: string, expiry: string, strike: number, right: OptionType, exchange?: string, currency?: string);
    secType: SecType;
    multiplier: number;
    get lastTradeDateOrContractMonth(): string;
}
export default Option;
