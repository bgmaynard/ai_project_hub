import OptionType from "../data/enum/option-type";
import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * A Future Option Contract
 */
export declare class FOP implements Contract {
    symbol: string;
    expiry: string;
    strike: number;
    right: OptionType;
    multiplier?: number;
    exchange?: string;
    currency?: string;
    constructor(symbol: string, expiry: string, strike: number, right: OptionType, multiplier?: number, exchange?: string, currency?: string);
    secType: SecType;
}
export default FOP;
