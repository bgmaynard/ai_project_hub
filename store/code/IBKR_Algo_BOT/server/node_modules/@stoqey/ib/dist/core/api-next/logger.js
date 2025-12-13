"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextLogger = void 0;
const __1 = require("../..");
/**
 * @internal
 *
 * The logger proxy to filter log levels.
 */
class IBApiNextLogger {
    constructor(logger) {
        this.logger = logger;
        /** The current log level */
        this._logLevel = __1.LogLevel.SYSTEM;
    }
    /** Get the current log level. */
    get logLevel() {
        return this._logLevel;
    }
    /** Set the current log level. */
    set logLevel(level) {
        this._logLevel = level;
    }
    /** Log a debug information. */
    debug(tag, args) {
        if (this._logLevel >= __1.LogLevel.DETAIL) {
            this.logger.debug(tag, args);
        }
    }
    /** Log a generic information. */
    info(tag, args) {
        if (this._logLevel >= __1.LogLevel.INFO) {
            this.logger.info(tag, args);
        }
    }
    /** Log a warning. */
    warn(tag, args) {
        if (this._logLevel >= __1.LogLevel.WARN) {
            this.logger.warn(tag, args);
        }
    }
    /** Log an error. */
    error(tag, args) {
        if (this._logLevel >= __1.LogLevel.ERROR) {
            this.logger.error(tag, args);
        }
    }
}
exports.IBApiNextLogger = IBApiNextLogger;
//# sourceMappingURL=logger.js.map