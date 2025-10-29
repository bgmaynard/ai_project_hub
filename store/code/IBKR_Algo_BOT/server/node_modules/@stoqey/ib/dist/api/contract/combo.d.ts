import SecType from "../data/enum/sec-type";
import ComboLeg from "./comboLeg";
import { Contract } from "./contract";
/**
 * A Combo contract.
 */
export declare class Combo implements Contract {
    symbol: string;
    comboLegs: ComboLeg[];
    currency?: string;
    exchange?: string;
    constructor(symbol: string, comboLegs: ComboLeg[], currency?: string, exchange?: string);
    secType: SecType;
}
export default Combo;
