import { Subscription } from "rxjs";
import { IBApiNext } from "../../api-next";
import Contract from "../../api/contract/contract";
import LogLevel from "../../api/data/enum/log-level";
/**
 * Base-class for the [[IBApiNext]] apps.
 */
export declare class IBApiNextApp {
    static readonly DEFAULT_CONTRACT_OPTIONS: [string, string][];
    static readonly DEFAULT_OPT_CONTRACT_OPTIONS: [string, string][];
    protected logLevel: LogLevel;
    constructor(appDescription: string, usageDescription: string, optionArgumentDescriptions: [string, string][], usageExample: string);
    /** Common command line options of all [[IBApiNext]] apps. */
    private readonly COMMON_OPTION_ARGUMENTS;
    /** The [[IBApiNext]] instance. */
    protected api: IBApiNext;
    /** The subscription on the IBApi errors. */
    protected error$: Subscription;
    /** The command-line arguments. */
    protected cmdLineArgs: Record<string, string | number>;
    /** Connect to TWS. */
    connect(reconnectInterval?: number, clientId?: number): void;
    /**
     * Print text to console.
     */
    printText(text: string): void;
    /**
     * Print an object (JSON formatted) to console.
     */
    printObject(obj: unknown): void;
    /**
     * Print an error message and exit the app with error code, unless -watch argument is present.
     */
    error(text: string): void;
    /**
     * Print a warning message
     */
    warn(text: string): void;
    /**
     * Print an wainformation message
     */
    info(text: string): void;
    /**
     * Print an wainformation message
     */
    debug(text: string): void;
    /**
     * Exit the app.
     */
    exit(exitCode?: number): void;
    /** Parse the command line. */
    private parseCommandLine;
    /** Format the help text. */
    private formatHelpText;
    /** get contract from command line args */
    getContractArg(): Contract;
    /** app startup */
    start(): void;
}
