/**
 * A WshEventData event.
 */
export declare class WshEventData {
    conId: number;
    fillWatchlist: boolean;
    fillPortfolio: boolean;
    fillCompetitors: boolean;
    startDate: string;
    endDate: string;
    totalLimit: number;
    constructor(conId: number, fillWatchlist?: boolean, fillPortfolio?: boolean, fillCompetitors?: boolean, startDate?: string, endDate?: string, totalLimit?: number);
    filter: string;
}
export default WshEventData;
