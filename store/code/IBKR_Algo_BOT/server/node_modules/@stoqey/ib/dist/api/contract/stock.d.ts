import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * Stock contract.
 */
export declare class Stock implements Contract {
    symbol: string;
    exchange?: string;
    currency?: string;
    constructor(symbol: string, exchange?: string, currency?: string);
    secType: SecType;
}
export default Stock;
