"use strict";
/**
 * This App will print histogram data of a contract.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const duration_unit_1 = __importDefault(require("../api/data/enum/duration-unit"));
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Print histogram data of a contract.";
const USAGE_TEXT = "Usage: histogram-data.js <options>";
const OPTION_ARGUMENTS = [
    [
        "conid=<number>",
        "(required) Contract ID (conId) of contract to receive histogram data for.",
    ],
    ["exchange=<name>", "The destination exchange name."],
    ["period=<seconds>", "(required) Period of which data is being requested"],
    [
        "periodUnit=<SECOND|DAY|WEEK|MONTH|YEAR>",
        "(required) Unit of the period argument",
    ],
];
const EXAMPLE_TEXT = "histogram-data.js -conid=3691937 -exchange=SMART -period=3 -periodUnit=DAY";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintHistogramDataApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        if (!this.cmdLineArgs.conid) {
            this.error("-conid argument missing.");
        }
        if (!this.cmdLineArgs.exchange) {
            this.error("-exchange argument missing.");
        }
        if (!this.cmdLineArgs.period) {
            this.error("-period argument missing.");
        }
        if (!this.cmdLineArgs.periodUnit) {
            this.error("-periodUnit argument missing.");
        }
        if (!(this.cmdLineArgs.periodUnit in duration_unit_1.default)) {
            this.error("Invalid -periodUnit argument value: " + this.cmdLineArgs.periodUnit);
        }
        this.api
            .getHistogramData({
            conId: this.cmdLineArgs.conid,
            exchange: this.cmdLineArgs.exchange,
        }, false, this.cmdLineArgs.period, this.cmdLineArgs.periodUnit)
            .then((data) => {
            this.printObject(data);
            this.exit();
        })
            .catch((err) => {
            this.error(`getHistogramData failed with '${err.error.message}'`);
        });
    }
}
// run the app
new PrintHistogramDataApp().start();
//# sourceMappingURL=histogram-data.js.map