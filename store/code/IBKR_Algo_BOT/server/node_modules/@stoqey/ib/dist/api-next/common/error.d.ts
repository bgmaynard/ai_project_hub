import { ErrorCode } from "../..";
/**
 * An error on the TWS / IB Gateway API or IBApiNext.
 */
export declare class IBApiNextError extends Error {
    /** The [[Error]] object. */
    readonly error: Error;
    /** The [[IBApi]] Error Code. */
    readonly code: ErrorCode;
    /**  The request id that caused the error, or -1. */
    readonly reqId: number;
    /** Additional information in case of order rejection */
    readonly advancedOrderReject?: unknown;
    constructor(error: Error, code: ErrorCode, reqId?: number, advancedOrderReject?: unknown);
}
