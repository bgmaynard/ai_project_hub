"use strict";
/**
 * This App will print the timestamp of earliest available historical data for a contract.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("..");
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// Help text and command line parsing                                          //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the timestamp of earliest available historical data for a contract.";
const USAGE_TEXT = "Usage: get-head-timestamp.js <options>";
const OPTION_ARGUMENTS = [
    ...ib_api_next_app_1.IBApiNextApp.DEFAULT_CONTRACT_OPTIONS,
];
const EXAMPLE_TEXT = "get-head-timestamp.js -symbol=AMZN -sectype=STK -currency=USD -exchange=SMART -conid=3691937 -port=4002";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintHeadTimestampApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        // print next unused order id
        this.api
            .getHeadTimestamp(this.getContractArg(), __1.WhatToShow.TRADES, true, 1)
            .then((timestamp) => {
            this.printText(timestamp);
            this.stop();
        })
            .catch((err) => {
            this.error(`getHeadTimestamp failed with '${err.error.message}'`);
        });
    }
    /**
     * Stop the app with success code.
     */
    stop() {
        this.exit();
    }
}
// run the app
new PrintHeadTimestampApp().start();
//# sourceMappingURL=get-head-timestamp.js.map