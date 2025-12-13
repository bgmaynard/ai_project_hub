"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Socket = exports.ConnectionStatus = void 0;
const net_1 = __importDefault(require("net"));
const util_1 = require("util");
const api_1 = require("../../api/api");
const event_name_1 = require("../../api/data/enum/event-name");
const min_server_version_1 = __importDefault(require("../../api/data/enum/min-server-version"));
const configuration_1 = __importDefault(require("../../common/configuration"));
const errorCode_1 = require("../../common/errorCode");
const encoder_1 = require("./encoder");
/**
 * @hidden
 * envelope encoding, applicable to useV100Plus mode only
 */
const MIN_VERSION_V100 = 100;
/**
 * @hidden
 * max message size, taken from Java client, applicable to useV100Plus mode only
 */
const MAX_V100_MESSAGE_LENGTH = 0xffffff;
/** @hidden */
const EOL = "\0";
/**
 * @hidden
 * add a delay after connect before sending commands
 */
// const CONNECT_DELAY = 600;
exports.ConnectionStatus = {
    Disconnected: 0,
    Disconnecting: 1,
    Connecting: 2,
    Connected: 3,
};
/**
 * @internal
 *
 * This class implements low-level details on the communication protocol of the
 * TWS/IB Gateway API server.
 */
class Socket {
    /**
     * Create a new [[Socket]] object.
     *
     * @param controller The parent [[Controller]] object.
     * @param options The API creation options.
     */
    constructor(controller, options = {}) {
        this.controller = controller;
        this.options = options;
        /** `connected` if the TCP socket is connected and [[OUT_MSG_ID.START_API]] has been sent.  */
        this._status = exports.ConnectionStatus.Disconnected;
        /** The IB API Server version, or 0 if not connected yet. */
        this._serverVersion = 0;
        /** The server connection time. */
        this._serverConnectionTime = "";
        /** Data fragment accumulation buffer. */
        this.dataFragment = "";
        /** `true` if no message from server has been received yet, `false` otherwise. */
        this.neverReceived = true;
        /** `true` if waiting for completion of an async operation, `false` otherwise.  */
        this.waitingAsync = false;
        /** `true` if V!00Pls protocol shall be used, `false` otherwise.  */
        this.useV100Plus = true;
        /** Accumulation buffer for fragmented V100 messages */
        this._v100MessageBuffer = Buffer.alloc(0);
        this._clientId =
            this.options.clientId !== undefined
                ? Math.floor(this.options.clientId)
                : configuration_1.default.default_client_id;
        // this.options.host = this.options.host;
        // this.options.port = this.options.port;
    }
    /** Returns `true` if connected to TWS/IB Gateway, `false` otherwise.  */
    get connected() {
        return this._status === exports.ConnectionStatus.Connected;
    }
    /** Returns connection status */
    get status() {
        return this._status;
    }
    /** Returns the IB API Server version. */
    get serverVersion() {
        return this._serverVersion;
    }
    /** The server connection time. */
    get serverConnectionTime() {
        return this._serverConnectionTime;
    }
    /** Get the current client id. */
    get clientId() {
        return this._clientId;
    }
    /**
     * Disable usage of V100Plus protocol.
     */
    disableUseV100Plus() {
        this.useV100Plus = false;
    }
    /**
     * Connect to the API server.
     *
     * @param clientId A unique client id (per TWS or IB Gateway instance).
     * When not specified, the client from [[IBApiCreationOptions]] or the
     * default client id (0) will used.
     */
    connect(clientId) {
        // Reject any connect attempt is not disconnected
        if (this._status >= exports.ConnectionStatus.Connecting)
            return;
        this._status = exports.ConnectionStatus.Connecting;
        // update client id
        if (clientId !== undefined) {
            this._clientId = Math.floor(clientId);
        }
        // pause controller while API startup sequence
        this.controller.pause();
        // reset state
        this.dataFragment = "";
        this.neverReceived = true;
        this.waitingAsync = false;
        this._v100MessageBuffer = Buffer.alloc(0);
        // create and connect TCP socket
        this.client = net_1.default
            .connect({
            host: this.options.host ?? configuration_1.default.ib_host,
            port: this.options.port ?? configuration_1.default.ib_port,
        }, () => this.onConnect())
            .on("data", (data) => this.onData(data))
            .on("close", () => this.onEnd())
            .on("end", () => this.onEnd())
            .on("error", (error) => this.onError(error));
    }
    /**
     * Disconnect from API server.
     */
    disconnect() {
        this._status = exports.ConnectionStatus.Disconnecting;
        // pause controller while connection is down.
        this.controller.pause();
        // disconnect TCP socket.
        this.client?.end();
        this.client?.destroy();
    }
    /**
     * Send tokens to API server.
     */
    send(tokens) {
        // flatten arrays and convert boolean types to 0/1
        tokens = this.flattenDeep(tokens);
        tokens.forEach((value, i) => {
            if (value === true || value === false || value instanceof Boolean) {
                tokens[i] = value ? 1 : 0;
            }
        });
        let stringData = tokens.join(EOL);
        if (this.useV100Plus) {
            let utf8Data;
            if (tokens[0] === "API\0") {
                // this is the initial API version message, which is special:
                // length is encoded after the 'API\0', followed by the actual tokens.
                const skip = 5; // 1 x 'API\0' token + 4 x length tokens
                stringData = tokens.slice(skip)[0];
                utf8Data = [
                    ...this.stringToUTF8Array(tokens[0]),
                    ...tokens.slice(1, skip),
                    ...this.stringToUTF8Array(stringData),
                ];
            }
            else {
                utf8Data = this.stringToUTF8Array(stringData);
            }
            // add length prefix only if not a string (strings use pre-V100 style)
            if (typeof tokens[0] !== "string") {
                utf8Data = [
                    ...this.numberTo32BitBigEndian(utf8Data.length + 1),
                    ...utf8Data,
                    0,
                ];
            }
            this.client?.write(Buffer.from(new Uint8Array(utf8Data)));
        }
        else {
            this.client?.write(stringData + EOL);
        }
        this.controller.emitEvent(event_name_1.EventName.sent, tokens, stringData);
    }
    /**
     * Called when data on the TCP socket has been arrived.
     */
    onData(data) {
        if (this.useV100Plus) {
            this._v100MessageBuffer = Buffer.concat([this._v100MessageBuffer, data]);
            if (this._v100MessageBuffer.length > MAX_V100_MESSAGE_LENGTH) {
                // At this point we have buffered enough data that we have exceeded the max known message length,
                // at which point this is likely an unrecoverable state and we should discard all prior data,
                // and disconnect the socket
                const size = this._v100MessageBuffer.length;
                this._v100MessageBuffer = Buffer.alloc(0);
                this.onError(new Error(`Message of size ${size} exceeded max message length ${MAX_V100_MESSAGE_LENGTH}`));
                this.disconnect();
                return;
            }
            while (this._v100MessageBuffer.length > 4) {
                const msgSize = this._v100MessageBuffer.readInt32BE();
                if (this._v100MessageBuffer.length >= 4 + msgSize) {
                    const segment = this._v100MessageBuffer.slice(4, 4 + msgSize);
                    this._v100MessageBuffer = this._v100MessageBuffer.slice(4 + msgSize);
                    this.onMessage(segment.toString("utf8"));
                }
                else {
                    // else keep data for later
                    return;
                }
            }
        }
        else {
            this.onMessage(data.toString());
        }
    }
    /**
     * Called when new tokens have been received from server.
     */
    onMessage(data) {
        // tokenize
        const dataWithFragment = this.dataFragment + data;
        let tokens = dataWithFragment.split(EOL);
        if (tokens[tokens.length - 1] !== "") {
            this.dataFragment = tokens[tokens.length - 1];
        }
        else {
            this.dataFragment = "";
        }
        tokens = tokens.slice(0, -1);
        this.controller.emitEvent(event_name_1.EventName.received, tokens.slice(0), data);
        // handle message data
        if (this.neverReceived) {
            // first message
            this.neverReceived = false;
            this.onServerVersion(tokens);
        }
        else {
            // post to queue
            if (this.useV100Plus) {
                this.controller.onMessage(tokens);
            }
            else {
                this.controller.onTokens(tokens);
            }
            // process queue
            this.controller.processIngressQueue();
        }
        // resume from async state
        if (this.waitingAsync) {
            this.waitingAsync = false;
            this.controller.resume();
        }
    }
    /**
     * Called when first data has arrived on the connection.
     */
    onServerVersion(tokens) {
        this._status = exports.ConnectionStatus.Connected;
        this._serverVersion = parseInt(tokens[0], 10);
        this._serverConnectionTime = tokens[1];
        if (this.useV100Plus &&
            (this._serverVersion < MIN_VERSION_V100 ||
                this._serverVersion > api_1.MAX_SUPPORTED_SERVER_VERSION)) {
            this.disconnect();
            this.controller.emitError(`Unsupported Version ${this._serverVersion}`, errorCode_1.ErrorCode.UNSUPPORTED_VERSION);
            return;
        }
        if (this._serverVersion < api_1.MIN_SERVER_VER_SUPPORTED) {
            this.disconnect();
            this.controller.emitError("The TWS is out of date and must be upgraded.", errorCode_1.ErrorCode.UPDATE_TWS);
            return;
        }
        this.startAPI();
        this.controller.emitEvent(event_name_1.EventName.connected);
        this.controller.emitEvent(event_name_1.EventName.server, this.serverVersion, this.serverConnectionTime);
    }
    /**
     * Start the TWS/IB Gateway API.
     */
    startAPI() {
        // start API
        const VERSION = 2;
        if (this.serverVersion >= 3) {
            if (this.serverVersion < min_server_version_1.default.LINKING) {
                this.send([this._clientId]);
            }
            else {
                if (this.serverVersion >= min_server_version_1.default.OPTIONAL_CAPABILITIES) {
                    this.send([encoder_1.OUT_MSG_ID.START_API, VERSION, this._clientId, ""]);
                }
                else {
                    this.send([encoder_1.OUT_MSG_ID.START_API, VERSION, this._clientId]);
                }
            }
        }
        // resume controller moved to crontroller
        // setTimeout(() => {
        //   this.controller.resume();
        // }, CONNECT_DELAY);
    }
    /**
     * Called when TCP socket has been connected.
     */
    onConnect() {
        // send client version (unless Version > 100)
        if (!this.useV100Plus) {
            this.send([configuration_1.default.client_version]);
            this.send([this._clientId]);
        }
        else {
            // Switch to GW API (Version 100+ requires length prefix)
            const config = this.buildVersionString(MIN_VERSION_V100, api_1.MAX_SUPPORTED_SERVER_VERSION);
            // config = config + connectOptions --- connectOptions are for IB internal use only: not supported
            this.send([
                "API\0",
                ...this.numberTo32BitBigEndian(config.length),
                config,
            ]);
        }
    }
    /**
     * Called when TCP socket connection has been closed.
     */
    onEnd() {
        if (this._status) {
            this._status = exports.ConnectionStatus.Disconnected;
            this.controller.emitEvent(event_name_1.EventName.disconnected);
        }
        this.controller.pause();
    }
    /**
     * Called when an error occurred on the TCP socket connection.
     */
    onError(err) {
        this.controller.emitError(err.message, errorCode_1.ErrorCode.CONNECT_FAIL);
    }
    /**
     * Build a V100Plus API version string.
     */
    buildVersionString(minVersion, maxVersion) {
        return ("v" +
            (minVersion < maxVersion ? minVersion + ".." + maxVersion : minVersion));
    }
    /**
     * Convert a (integer) number to a 4-byte big endian byte array.
     */
    numberTo32BitBigEndian(val) {
        const result = new Array(4);
        let pos = 0;
        result[pos++] = 0xff & (val >> 24);
        result[pos++] = 0xff & (val >> 16);
        result[pos++] = 0xff & (val >> 8);
        result[pos++] = 0xff & val;
        return result;
    }
    /**
     * Encode a string to a UTF8 byte array.
     */
    stringToUTF8Array(val) {
        return Array.from(new util_1.TextEncoder().encode(val));
    }
    /**
     * Flatten an array.
     *
     * Also works for nested arrays (i.e. arrays inside arrays inside arrays)
     */
    flattenDeep(arr, result = []) {
        for (let i = 0, length = arr.length; i < length; i++) {
            const value = arr[i];
            if (Array.isArray(value)) {
                this.flattenDeep(value, result);
            }
            else {
                result.push(value);
            }
        }
        return result;
    }
}
exports.Socket = Socket;
//# sourceMappingURL=socket.js.map