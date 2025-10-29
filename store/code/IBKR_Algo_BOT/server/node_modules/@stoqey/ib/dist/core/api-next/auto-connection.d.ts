import { Observable } from "rxjs";
import IBApi, { ConnectionState, IBApiCreationOptions } from "../..";
import { Logger } from "../../api-next/common/logger";
/**
 * @internal
 *
 * This class implements auto re-connection for the [[IBApi]].
 *
 * It will monitor the connection state and poll the TWS / IB Gateway at
 * regular intervals to also detect abnormal connection drops.
 * If a connection drop is detected, a new connection will be initiated
 * after the specified reconnection interval.
 */
export declare class IBApiAutoConnection extends IBApi {
    readonly reconnectInterval: number;
    private readonly watchdogInterval;
    private readonly logger;
    readonly options?: IBApiCreationOptions;
    /**
     * Create an [[IBApiAutoConnection]] object.
     *
     * @param reconnectInterval The auto-reconnect interval in milliseconds.
     * Use 0 to disable auto re-connect.
     * @param reconnectInterval The watchdog interval in milliseconds.
     * Use 0 to disable auto re-connect.
     * @param options [[IBApi]] Creation options.
     * @param logger The [[IBApiNextLogger]] logger instance ot receive log messages
     */
    constructor(reconnectInterval: number, watchdogInterval: number, logger: Logger, options?: IBApiCreationOptions);
    /**
     * If defined, this is the client id that will be used on all
     * re-connection attempt. If undefined [[currentClientId]] will
     * be used and incremented on each attempt.
     */
    private fixedClientId?;
    /** The current client id. */
    private currentClientId;
    /** true if auto re-connect is enabled, false otherwise. */
    private autoReconnectEnabled;
    /** The auto re-connect timeout. */
    private reconnectionTimeout?;
    /** The connection-watchdog timeout. */
    private connectionWatchdogTimeout?;
    /** Ingress timestamp of last received message data from TWS. */
    private lastDataIngressTm?;
    /** The connection-state [[BehaviorSubject]]. */
    private readonly _connectionState;
    /** Get the connection-state as an [[Observable]]. */
    get connectionState(): Observable<ConnectionState>;
    /**
     * Connect to the TWS or IB Gateway.
     *
     * @param clientId A fixed client id to be used on all connection
     * attempts. If not specified, the first connection will use the
     * default client id (0) and increment it with each re-connection
     * attempt.
     *
     * @sse [[connectionState]] for observing the connection state.
     */
    connect(clientId?: number): IBApi;
    /**
     * Disconnect from the TWS or IB Gateway.
     *
     * Use [[connectionState]] for observing the connection state.
     */
    disconnect(): IBApi;
    /**
     * Called when [[EventName.connected]] event has been received.
     */
    private onConnected;
    /**
     * Re-establish the connection.
     */
    private reConnect;
    /**
     * Start the re-connection timer.
     */
    private runReConnectTimer;
    /**
     * Stop the re-connection timer.
     */
    private stopReConnectTimer;
    /**
     * Start the connection watchdog
     */
    private runWatchdog;
    /**
     * Stop the connection watchdog.
     */
    private stopWatchdog;
    /**
     * Called when an [[EventName.disconnected]] event has been received,
     * or the connection-watchdog has detected a dead connection.
     */
    onDisconnected(): void;
}
