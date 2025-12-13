"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConnectionState = void 0;
/**
 * Status of the connection to TWS / IB Gateway.
 */
var ConnectionState;
(function (ConnectionState) {
    /** Disconnected from TWS / IB Gateway. */
    ConnectionState[ConnectionState["Disconnected"] = 0] = "Disconnected";
    /** Current connecting to TWS / IB Gateway. */
    ConnectionState[ConnectionState["Connecting"] = 1] = "Connecting";
    /** Connected to TWS / IB Gateway. */
    ConnectionState[ConnectionState["Connected"] = 2] = "Connected";
})(ConnectionState || (exports.ConnectionState = ConnectionState = {}));
//# sourceMappingURL=connection-state.js.map