import SecType from "../data/enum/sec-type";
import { Contract } from "./contract";
/**
 * Index contract.
 */
export declare class Index implements Contract {
    symbol: string;
    currency?: string;
    exchange?: string;
    constructor(symbol: string, currency?: string, exchange?: string);
    secType: SecType;
}
export default Index;
