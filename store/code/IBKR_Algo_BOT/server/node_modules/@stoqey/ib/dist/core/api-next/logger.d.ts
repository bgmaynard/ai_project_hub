import { LogLevel } from "../..";
import { Logger } from "../../api-next/common/logger";
/**
 * @internal
 *
 * The logger proxy to filter log levels.
 */
export declare class IBApiNextLogger {
    private logger;
    constructor(logger: Logger);
    /** The current log level */
    private _logLevel;
    /** Get the current log level. */
    get logLevel(): LogLevel;
    /** Set the current log level. */
    set logLevel(level: LogLevel);
    /** Log a debug information. */
    debug(tag: string, args: unknown[] | string): void;
    /** Log a generic information. */
    info(tag: string, args: unknown[] | string): void;
    /** Log a warning. */
    warn(tag: string, args: unknown[] | string): void;
    /** Log an error. */
    error(tag: string, args: unknown[] | string): void;
}
