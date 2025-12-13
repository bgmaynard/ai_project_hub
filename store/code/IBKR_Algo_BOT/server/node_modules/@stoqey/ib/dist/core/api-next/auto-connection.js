"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiAutoConnection = void 0;
const rxjs_1 = require("rxjs");
const __1 = __importStar(require("../.."));
/** The log tag. */
const LOG_TAG = "IBApiAutoConnection";
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
class IBApiAutoConnection extends __1.default {
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
    constructor(reconnectInterval, watchdogInterval, logger, options) {
        super(options);
        this.reconnectInterval = reconnectInterval;
        this.watchdogInterval = watchdogInterval;
        this.logger = logger;
        this.options = options;
        /** true if auto re-connect is enabled, false otherwise. */
        this.autoReconnectEnabled = true;
        /** The connection-state [[BehaviorSubject]]. */
        this._connectionState = new rxjs_1.BehaviorSubject(__1.ConnectionState.Disconnected);
        this.on(__1.EventName.connected, () => this.onConnected());
        this.on(__1.EventName.disconnected, () => this.onDisconnected());
        this.on(__1.EventName.received, () => (this.lastDataIngressTm = Date.now()));
        this.on(__1.EventName.error, (_error, code) => {
            if (code === __1.ErrorCode.CONNECT_FAIL) {
                this.onDisconnected();
            }
        });
        this.on(__1.EventName.currentTime, () => (this.lastDataIngressTm = Date.now()));
    }
    /** Get the connection-state as an [[Observable]]. */
    get connectionState() {
        return this._connectionState;
    }
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
    connect(clientId) {
        this.autoReconnectEnabled = true;
        this.fixedClientId = clientId;
        this.currentClientId =
            (clientId === undefined ? this.options?.clientId : clientId) ??
                Math.floor(Math.random() * 32766) + 1;
        if (this._connectionState.getValue() === __1.ConnectionState.Disconnected) {
            this._connectionState.next(__1.ConnectionState.Connecting);
            this.logger.info(LOG_TAG, `Connecting to TWS with client id ${this.currentClientId}`);
            super.connect(this.currentClientId);
        }
        return this;
    }
    /**
     * Disconnect from the TWS or IB Gateway.
     *
     * Use [[connectionState]] for observing the connection state.
     */
    disconnect() {
        this.autoReconnectEnabled = false;
        if (this._connectionState.getValue() !== __1.ConnectionState.Disconnected) {
            this.logger.info(LOG_TAG, `Disconnecting client id ${this.currentClientId} from TWS.`);
            this._connectionState.next(__1.ConnectionState.Disconnected);
            if (this.isConnected) {
                super.disconnect();
            }
        }
        return this;
    }
    /**
     * Called when [[EventName.connected]] event has been received.
     */
    onConnected() {
        if (this._connectionState.getValue() !== __1.ConnectionState.Connected) {
            // signal connect state
            this._connectionState.next(__1.ConnectionState.Connected);
            this.logger.info(LOG_TAG, `Successfully connected to TWS with client id ${this.currentClientId}.`);
            // cancel reconnect timer and run the connection watchdog
            this.stopReConnectTimer();
            this.runWatchdog();
        }
    }
    /**
     * Re-establish the connection.
     */
    reConnect() {
        // verify and update state
        if (this._connectionState.getValue() !== __1.ConnectionState.Disconnected ||
            !this.autoReconnectEnabled) {
            return;
        }
        this._connectionState.next(__1.ConnectionState.Connecting);
        // connect to IB
        this.currentClientId =
            this.fixedClientId !== undefined
                ? this.fixedClientId
                : this.currentClientId + 1;
        this.logger.info(LOG_TAG, `Re-Connecting to TWS with client id ${this.currentClientId}`);
        super.disconnect();
        super.connect(this.currentClientId);
    }
    /**
     * Start the re-connection timer.
     */
    runReConnectTimer() {
        // verify state
        if (!this.reconnectInterval || !this.autoReconnectEnabled) {
            return;
        }
        this.logger.info(LOG_TAG, `Re-Connecting to TWS in ${this.reconnectInterval / 1000}s...`);
        if (this.reconnectionTimeout) {
            clearTimeout(this.reconnectionTimeout);
        }
        this.reconnectionTimeout = setTimeout(() => {
            this.reConnect();
        }, this.reconnectInterval);
    }
    /**
     * Stop the re-connection timer.
     */
    stopReConnectTimer() {
        // verify state
        if (this.reconnectionTimeout === undefined) {
            return;
        }
        // reset timeout
        clearTimeout(this.reconnectionTimeout);
        delete this.reconnectionTimeout;
    }
    /**
     * Start the connection watchdog
     */
    runWatchdog() {
        // verify state
        if (!this.watchdogInterval || this.connectionWatchdogTimeout) {
            return;
        }
        // run watchdog
        this.logger.debug(LOG_TAG, `Starting connection watchdog with ${this.watchdogInterval}ms interval.`);
        this.connectionWatchdogTimeout = setInterval(() => {
            let triggerReconnect = false;
            if (this.lastDataIngressTm === undefined) {
                triggerReconnect = true;
            }
            else {
                const elapsed = Date.now() - this.lastDataIngressTm;
                if (elapsed > this.watchdogInterval) {
                    triggerReconnect = true;
                }
            }
            if (triggerReconnect && !this.isConnected) {
                this.logger.debug(LOG_TAG, "Connection watchdog timeout. Dropping connection.");
                this.onDisconnected();
            }
            // trigger at least some message if connection is idle
            this.reqCurrentTime();
        }, this.watchdogInterval / 2);
    }
    /**
     * Stop the connection watchdog.
     */
    stopWatchdog() {
        // verify state
        if (this.connectionWatchdogTimeout === undefined) {
            return;
        }
        // reset interval
        clearInterval(this.connectionWatchdogTimeout);
        delete this.connectionWatchdogTimeout;
    }
    /**
     * Called when an [[EventName.disconnected]] event has been received,
     * or the connection-watchdog has detected a dead connection.
     */
    onDisconnected() {
        this.logger.debug(LOG_TAG, "onDisconnected()");
        // verify state and update state
        if (this.isConnected) {
            this.logger.debug(LOG_TAG, `Disconnecting client id ${this.currentClientId} from TWS (state-sync).`);
            this.disconnect();
            this.autoReconnectEnabled = true;
        }
        if (this._connectionState.getValue() !== __1.ConnectionState.Disconnected) {
            this._connectionState.next(__1.ConnectionState.Disconnected);
        }
        // stop watch and run re-connect timer
        this.stopWatchdog();
        this.runReConnectTimer();
    }
}
exports.IBApiAutoConnection = IBApiAutoConnection;
//# sourceMappingURL=auto-connection.js.map