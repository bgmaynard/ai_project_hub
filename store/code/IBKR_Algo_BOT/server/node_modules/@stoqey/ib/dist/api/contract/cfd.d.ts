import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * A CFD contract.
 */
export declare class CFD implements Contract {
    symbol: string;
    currency?: string;
    exchange?: string;
    constructor(symbol: string, currency?: string, exchange?: string);
    secType: SecType;
}
export default CFD;
