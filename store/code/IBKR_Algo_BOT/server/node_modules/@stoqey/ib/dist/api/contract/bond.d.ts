import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * A Bond Contract
 */
export declare class Bond implements Contract {
    symbol: string;
    maturity?: string;
    exchange?: string;
    currency?: string;
    constructor(symbol: string, maturity?: string, exchange?: string, currency?: string);
    secType: SecType;
    get lastTradeDateOrContractMonth(): string;
}
export default Bond;
