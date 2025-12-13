"use strict";
/**
 * This App will print real time market data of a given contract id.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const api_next_1 = require("../api-next");
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Print real time market data of a given contract id.";
const USAGE_TEXT = "Usage: market-data.js <options>";
const OPTION_ARGUMENTS = [
    ...ib_api_next_app_1.IBApiNextApp.DEFAULT_CONTRACT_OPTIONS,
    ["ticks=<ticks>", "Comma separated list of generic ticks to fetch."],
];
const EXAMPLE_TEXT = "market-data.js -symbol=AMZN -sectype=STK -exchange=SMART -conid=3691937";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintMarketDataApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.subscription$ = this.api
            .getMarketData(this.getContractArg(), this.cmdLineArgs.ticks, false, false)
            .subscribe({
            next: (marketData) => {
                const changedOrAddedDataWithTickNames = new Map();
                marketData.added?.forEach((tick, type) => {
                    if (type > api_next_1.IBApiNextTickType.API_NEXT_FIRST_TICK_ID) {
                        changedOrAddedDataWithTickNames.set(api_next_1.IBApiNextTickType[type], tick.value);
                    }
                    else {
                        changedOrAddedDataWithTickNames.set(api_next_1.IBApiTickType[type], tick.value);
                    }
                });
                marketData.changed?.forEach((tick, type) => {
                    if (type > api_next_1.IBApiNextTickType.API_NEXT_FIRST_TICK_ID) {
                        changedOrAddedDataWithTickNames.set(api_next_1.IBApiNextTickType[type], tick.value);
                    }
                    else {
                        changedOrAddedDataWithTickNames.set(api_next_1.IBApiTickType[type], tick.value);
                    }
                });
                this.printObject(changedOrAddedDataWithTickNames);
            },
            error: (err) => {
                this.subscription$?.unsubscribe();
                this.error(`getMarketData failed with '${err.error.message}' (${err.code})`);
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
new PrintMarketDataApp().start();
//# sourceMappingURL=market-data.js.map