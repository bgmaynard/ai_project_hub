import { IBApiCreationOptions } from "../../api/api";
import { Controller } from "./controller";
/**
 * @hidden
 * add a delay after connect before sending commands
 */
export declare const ConnectionStatus: {
    readonly Disconnected: 0;
    readonly Disconnecting: 1;
    readonly Connecting: 2;
    readonly Connected: 3;
};
export type ConnectionStatus = (typeof ConnectionStatus)[keyof typeof ConnectionStatus];
/**
 * @internal
 *
 * This class implements low-level details on the communication protocol of the
 * TWS/IB Gateway API server.
 */
export declare class Socket {
    private controller;
    private options;
    /**
     * Create a new [[Socket]] object.
     *
     * @param controller The parent [[Controller]] object.
     * @param options The API creation options.
     */
    constructor(controller: Controller, options?: IBApiCreationOptions);
    /** The TCP client socket. */
    private client?;
    /** `connected` if the TCP socket is connected and [[OUT_MSG_ID.START_API]] has been sent.  */
    private _status;
    /** The IB API Server version, or 0 if not connected yet. */
    private _serverVersion;
    /** The server connection time. */
    private _serverConnectionTime;
    /** Data fragment accumulation buffer. */
    private dataFragment;
    /** `true` if no message from server has been received yet, `false` otherwise. */
    private neverReceived;
    /** `true` if waiting for completion of an async operation, `false` otherwise.  */
    private waitingAsync;
    /** `true` if V!00Pls protocol shall be used, `false` otherwise.  */
    private useV100Plus;
    /** Accumulation buffer for fragmented V100 messages */
    private _v100MessageBuffer;
    /** The current client id. */
    private _clientId;
    /** Returns `true` if connected to TWS/IB Gateway, `false` otherwise.  */
    get connected(): boolean;
    /** Returns connection status */
    get status(): ConnectionStatus;
    /** Returns the IB API Server version. */
    get serverVersion(): number;
    /** The server connection time. */
    get serverConnectionTime(): string;
    /** Get the current client id. */
    get clientId(): number;
    /**
     * Disable usage of V100Plus protocol.
     */
    disableUseV100Plus(): void;
    /**
     * Connect to the API server.
     *
     * @param clientId A unique client id (per TWS or IB Gateway instance).
     * When not specified, the client from [[IBApiCreationOptions]] or the
     * default client id (0) will used.
     */
    connect(clientId?: number): void;
    /**
     * Disconnect from API server.
     */
    disconnect(): void;
    /**
     * Send tokens to API server.
     */
    send(tokens: unknown[]): void;
    /**
     * Called when data on the TCP socket has been arrived.
     */
    private onData;
    /**
     * Called when new tokens have been received from server.
     */
    private onMessage;
    /**
     * Called when first data has arrived on the connection.
     */
    private onServerVersion;
    /**
     * Start the TWS/IB Gateway API.
     */
    private startAPI;
    /**
     * Called when TCP socket has been connected.
     */
    private onConnect;
    /**
     * Called when TCP socket connection has been closed.
     */
    private onEnd;
    /**
     * Called when an error occurred on the TCP socket connection.
     */
    private onError;
    /**
     * Build a V100Plus API version string.
     */
    private buildVersionString;
    /**
     * Convert a (integer) number to a 4-byte big endian byte array.
     */
    private numberTo32BitBigEndian;
    /**
     * Encode a string to a UTF8 byte array.
     */
    private stringToUTF8Array;
    /**
     * Flatten an array.
     *
     * Also works for nested arrays (i.e. arrays inside arrays inside arrays)
     */
    private flattenDeep;
}
