import { IBApi, IBApiCreationOptions } from "../../api/api";
import { EventName } from "../../api/data/enum/event-name";
import { Decoder, DecoderCallbacks } from "./decoder";
import { Encoder, EncoderCallbacks } from "./encoder";
/**
 * @internal
 *
 * This class implements the dispatcher between public API and the
 * underlying I/O code.
 */
export declare class Controller implements EncoderCallbacks, DecoderCallbacks {
    private ib;
    private options?;
    /**
     *
     * @param ib The [[IBApi]] object.
     * @param _options The [[IBApi]] creation options.
     */
    constructor(ib: IBApi, options?: IBApiCreationOptions);
    /** The API socket object. */
    private socket;
    /** The command buffer. */
    private readonly commands;
    /** The rate limiter function. */
    private readonly rateLimiter;
    /** The API message encoder. */
    readonly encoder: Encoder;
    /** The API message decoder. */
    readonly decoder: Decoder;
    /**
     * Pause command processing.
     */
    pause(): void;
    /**
     * Resume command processing.
     */
    resume(): void;
    /**
     * Connect to the API server.
     */
    connect(clientId?: number): void;
    /**
     * Disconnect from the API server.
     */
    disconnect(): void;
    /**
     * Schedule an API command for sending.
     *
     * @param funcName API function name.
     * @param data Array of tokens to send.
     */
    schedule(func: () => void): void;
    /**
     * Send an array of tokens to the sever immediately.
     *
     * @param data Array of tokens to send.
     */
    send(...args: unknown[]): void;
    /**
     * Progress the ingress data queue.
     */
    processIngressQueue(): void;
    /**
     * Called when a message has been arrived on the API server connection.
     *
     * Used on V100 protocol.
     */
    onMessage(tokens: string[]): void;
    /**
     * Called when a message has been arrived on the API server connection.
     *
     * Used on pre-V100 protocol.
     */
    onTokens(tokens: string[]): void;
    /**
     * Get the API server version.
     *
     * This function is called from the [[Decoder]] and [[Encoder]]
     * (via [DecoderCallbacks.serverVersion] and [DecoderCallbacks.serverVersion]).
     */
    get serverVersion(): number;
    /**
     * Returns `true` if currently connected to server, `false` otherwise.
     */
    get connected(): boolean;
    /**
     * Disable usage of V100Plus protocol.
     */
    disableUseV100Plus(): void;
    /**
     * Send a message to the server connection.
     *
     * This function is called from the [[Encoder]] (via [EncoderCallbacks.sendMsg]).
     *
     * @param args Array of tokens to send.
     * Can contain nested arrays.
     */
    sendMsg(...tokens: unknown[]): void;
    /**
     * Emit an event to public API interface.
     *
     * This function is called from the [[Decoder]] (via [DecoderCallbacks.emitEvent]).
     *
     * @param eventName Event name.
     * @param args Event arguments.
     */
    emitEvent(eventName: EventName, ...args: unknown[]): void;
    /**
     * Emit an information message event to public API interface.
     *
     * This function is called from the [[Decoder]] (via [DecoderCallbacks.emitInfo]).
     *
     * @param message The message text.
     * @param code The message code.
     */
    emitInfo(message: string, code: number): void;
    /**
     * Emit an error event to public API interface.
     *
     * This function is called from the [[Decoder]] and [[Encoder]]
     * (via [DecoderCallbacks.emitError] and [DecoderCallbacks.emitError]).
     *
     * @param errMsg The error test message.
     * @param code The error code.
     * @param reqId RequestId associated to this error.
     * @param advancedOrderReject Additional error data (optional).
     */
    emitError(errMsg: string, code: number, reqId?: number, advancedOrderReject?: unknown): void;
    /**
     * Execute a command.
     *
     * @param callback Callback function to invoke.
     * @param data Command data.
     */
    private static execute;
    /**
     * Execute a connect command.
     *
     * @see [[connect]]
     */
    private executeConnect;
    /**
     * Execute a disconnect command.
     *
     * @see [[disconnect]]
     */
    private executeDisconnect;
    /**
     * Send raw token data to the server connection.
     *
     * @param tokens Array of tokens to send.
     *
     * @see [[send]]
     */
    private executeSend;
}
