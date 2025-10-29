/**
 * Status of the connection to TWS / IB Gateway.
 */
export declare enum ConnectionState {
    /** Disconnected from TWS / IB Gateway. */
    Disconnected = 0,
    /** Current connecting to TWS / IB Gateway. */
    Connecting = 1,
    /** Connected to TWS / IB Gateway. */
    Connected = 2
}
