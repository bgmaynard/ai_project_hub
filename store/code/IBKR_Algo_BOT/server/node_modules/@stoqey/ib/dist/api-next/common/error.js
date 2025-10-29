"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextError = void 0;
const __1 = require("../..");
/**
 * An error on the TWS / IB Gateway API or IBApiNext.
 */
class IBApiNextError extends Error {
    constructor(error, code, reqId = __1.ErrorCode.NO_VALID_ID, advancedOrderReject) {
        super(error.message); // Call the parent constructor
        this.name = "IBApiNextError"; // Set the error name
        Object.setPrototypeOf(this, IBApiNextError.prototype); // Ensure correct prototype chain
        this.error = error;
        this.code = code;
        this.reqId = reqId;
        this.advancedOrderReject = advancedOrderReject;
    }
}
exports.IBApiNextError = IBApiNextError;
//# sourceMappingURL=error.js.map