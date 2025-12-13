"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextApp = void 0;
const path_1 = __importDefault(require("path"));
const api_next_1 = require("../../api-next");
const log_level_1 = __importDefault(require("../../api/data/enum/log-level"));
const configuration_1 = __importDefault(require("../../common/configuration"));
const errorCode_1 = require("../../common/errorCode");
const logger_1 = __importDefault(require("../../common/logger"));
/**
 * @internal
 *
 * JSON replace function to convert ES6 Maps to tuple arrays.
 */
function jsonReplacer(key, value) {
    if (value instanceof Map) {
        const tuples = [];
        value.forEach((v, k) => {
            tuples.push([k, v]);
        });
        return tuples;
    }
    else {
        return value;
    }
}
/**
 * Base-class for the [[IBApiNext]] apps.
 */
class IBApiNextApp {
    constructor(appDescription, usageDescription, optionArgumentDescriptions, usageExample) {
        /** Common command line options of all [[IBApiNext]] apps. */
        this.COMMON_OPTION_ARGUMENTS = [
            ["h", "(or -help) Print the help text."],
            ["log=<log_level>", "Log level. Valid values: error, warn, info, debug."],
            [
                "host=<hostname>",
                "IP or hostname of the TWS or IB Gateway. Default is 127.0.0.1.",
            ],
            ["port=<number>", "Post number of the TWS or IB Gateway. Default is 4002."],
            [
                "clientId=<number>",
                "Client id of current ib connection. Default is random",
            ],
            [
                "watch",
                "Watch for changes. If specified, the app will keep running and print updates as received from TWS. " +
                    "If not specified, the app will print a one-time snapshot and then exit.",
            ],
        ];
        this.parseCommandLine(appDescription, usageDescription, [...this.COMMON_OPTION_ARGUMENTS, ...optionArgumentDescriptions], usageExample);
        if (this.cmdLineArgs.log) {
            switch (this.cmdLineArgs.log) {
                case "error":
                    this.logLevel = log_level_1.default.ERROR;
                    break;
                case "warn":
                    this.logLevel = log_level_1.default.WARN;
                    break;
                case "info":
                    this.logLevel = log_level_1.default.INFO;
                    break;
                case "debug":
                    this.logLevel = log_level_1.default.DETAIL;
                    break;
                default:
                    this.error(`Unknown value '${this.cmdLineArgs.log}' on -log argument.`);
                    break;
            }
        }
        else {
            this.logLevel = log_level_1.default.ERROR;
        }
    }
    /** Connect to TWS. */
    connect(reconnectInterval, clientId) {
        // create the IBApiNext object
        const port = this.cmdLineArgs.port ?? configuration_1.default.ib_port;
        const host = this.cmdLineArgs.host ?? configuration_1.default.ib_host;
        if (reconnectInterval === undefined && this.cmdLineArgs.watch)
            reconnectInterval = 10000;
        if (clientId === undefined && this.cmdLineArgs.clientId)
            clientId = +this.cmdLineArgs.clientId;
        this.info(`Logging into server: ${host}:${port}`);
        if (!this.api) {
            this.api = new api_next_1.IBApiNext({
                reconnectInterval,
                host,
                port,
                logger: logger_1.default,
            });
            this.api.logLevel = this.logLevel;
        }
        // log generic errors (reqId = -1) and exit with failure code
        if (!this.error$) {
            this.error$ = this.api.errorSubject.subscribe((error) => {
                if ((0, errorCode_1.isNonFatalError)(error.code, error.error)) {
                    this.warn(`${error.error.message} (Error #${error.code})`);
                }
                else {
                    this.error(`${error.error.message} (Error #${error.code}) ${error.advancedOrderReject ? error.advancedOrderReject : ""}`);
                }
            });
        }
        try {
            this.api.connect(clientId);
        }
        catch (error) {
            this.error(error.message);
        }
    }
    /**
     * Print text to console.
     */
    printText(text) {
        console.log(text);
    }
    /**
     * Print an object (JSON formatted) to console.
     */
    printObject(obj) {
        console.log(`${JSON.stringify(obj, jsonReplacer, 2)}`);
    }
    /**
     * Print an error message and exit the app with error code, unless -watch argument is present.
     */
    error(text) {
        logger_1.default.error(text);
        if (!this.cmdLineArgs.watch) {
            this.exit(1);
        }
    }
    /**
     * Print a warning message
     */
    warn(text) {
        if (this.logLevel >= log_level_1.default.WARN)
            logger_1.default.warn(text);
    }
    /**
     * Print an wainformation message
     */
    info(text) {
        if (this.logLevel >= log_level_1.default.INFO)
            logger_1.default.info(text);
    }
    /**
     * Print an wainformation message
     */
    debug(text) {
        if (this.logLevel >= log_level_1.default.DETAIL)
            logger_1.default.debug(text);
    }
    /**
     * Exit the app.
     */
    exit(exitCode = 0) {
        this.error$?.unsubscribe();
        process.exit(exitCode);
    }
    /** Parse the command line. */
    parseCommandLine(description, usage, optionArguments, example) {
        this.cmdLineArgs = {};
        process.argv.slice(2).forEach((arg) => {
            const pair = arg.split("=");
            const name = pair[0].substr(1);
            if (!optionArguments.find((v) => v[0].split("=")[0] == name)) {
                console.error("ERROR: Unknown argument " + pair[0]);
                this.exit(1);
            }
            this.cmdLineArgs[name] = pair.length > 1 ? (pair[1] ?? "1") : "1";
        });
        if (this.cmdLineArgs.h || this.cmdLineArgs.help) {
            console.info(this.formatHelpText(description, usage, optionArguments, example));
            process.exit(0);
        }
    }
    /** Format the help text. */
    formatHelpText(description, usage, options, example) {
        let result = description + "\n" + usage + "\n" + "Options:\n";
        options.forEach((argument) => {
            result += "  -" + argument[0] + ": " + argument[1] + "\n";
        });
        return result + "Example: " + example;
    }
    /** get contract from command line args */
    getContractArg() {
        return {
            conId: this.cmdLineArgs.conid ?? undefined,
            secType: this.cmdLineArgs.sectype,
            symbol: this.cmdLineArgs.symbol,
            localSymbol: this.cmdLineArgs.localsymbol,
            currency: this.cmdLineArgs.currency ?? "USD",
            exchange: this.cmdLineArgs.exchange ?? "SMART",
            lastTradeDateOrContractMonth: this.cmdLineArgs.expiry,
            strike: this.cmdLineArgs.strike ?? undefined,
            right: this.cmdLineArgs.right,
            multiplier: this.cmdLineArgs.multiplier ?? undefined,
        };
    }
    /** app startup */
    start() {
        const scriptName = path_1.default.basename(__filename);
        this.info(`Starting ${scriptName} script`);
        this.connect();
        this.api.setMarketDataType(api_next_1.MarketDataType.DELAYED_FROZEN);
    }
}
exports.IBApiNextApp = IBApiNextApp;
// private compat_mode: boolean = false;
IBApiNextApp.DEFAULT_CONTRACT_OPTIONS = [
    ["conid=<number>", "Contract ID (conId) of the contract."],
    [
        "sectype=<type>",
        "The security type. Valid values: STK, OPT, FUT, IND, FOP, CFD, CASH, BAG, BOND, CMDTY, NEWS and FUND",
    ],
    ["symbol=<name>", "The symbol name."],
    ["currency=<currency>", "The contract currency."],
    ["exchange=<name>", "The destination exchange name."],
    ["localsymbol=<name>", "The symbol's local symbol."],
];
IBApiNextApp.DEFAULT_OPT_CONTRACT_OPTIONS = [
    ...IBApiNextApp.DEFAULT_CONTRACT_OPTIONS,
    [
        "expiry=<YYYYMM>",
        "The contract's last trading day or contract month (for Options and Futures)." +
            "Strings with format YYYYMM will be interpreted as the Contract Month whereas YYYYMMDD will be interpreted as Last Trading Day.",
    ],
    ["strike=<number>", "The option's strike price."],
    ["right=<P|C>", " The option type. Valid values are P, PUT, C, CALL."],
    ["multiplier=<number>", "The option's multiplier."],
];
//# sourceMappingURL=ib-api-next-app.js.map