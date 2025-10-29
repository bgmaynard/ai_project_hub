"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const market_scanner_1 = require("../api-next/market-scanner/market-scanner");
const logger_1 = __importDefault(require("../common/logger"));
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Print most active stocks scan.";
const USAGE_TEXT = "Usage: market-scanner.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "market-scanner.js -conid=3691937 -exchange=SMART -period=3 -periodUnit=DAY";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintMarketScreenerApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.subscription$ = this.api
            .getMarketScanner({
            abovePrice: 1,
            scanCode: market_scanner_1.ScanCode.MOST_ACTIVE,
            locationCode: market_scanner_1.LocationCode.STK_US,
            instrument: market_scanner_1.Instrument.STK,
            numberOfRows: 20,
        })
            .subscribe({
            next: (data) => {
                this.printObject(data.all);
                if (!this.cmdLineArgs.watch)
                    this.stop();
            },
            error: (error) => {
                logger_1.default.error("Error from the subscriber", error);
                this.stop();
            },
        });
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        this.subscription$?.unsubscribe();
        this.exit();
    }
}
// run the app
new PrintMarketScreenerApp().start();
//# sourceMappingURL=market-scanner.js.map