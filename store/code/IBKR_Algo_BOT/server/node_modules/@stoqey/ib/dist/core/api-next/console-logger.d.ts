import { Logger, LogLevel } from "../..";
/**
 * @internal
 *
 * The logger implementation of [[IBApiNext]].
 */
export declare class ConsoleLogger implements Logger {
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
