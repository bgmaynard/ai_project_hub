import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * Crypto contract.
 */
export declare class Crypto implements Contract {
    symbol: string;
    exchange?: string;
    currency?: string;
    constructor(symbol: string, exchange?: string, currency?: string);
    secType: SecType;
}
export default Crypto;
