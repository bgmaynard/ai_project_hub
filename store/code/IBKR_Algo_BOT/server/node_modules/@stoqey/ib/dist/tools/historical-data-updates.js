"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const what_to_show_1 = require("../api/historical/what-to-show");
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Print historical chart data of a contract.";
const USAGE_TEXT = "Usage: historical-data.js <options>";
const OPTION_ARGUMENTS = [
    [
        "conid=<number>",
        "(required) Contract ID (conId) of contract to receive real-time bar updates for.",
    ],
    ["exchange=<name>", "The destination exchange name."],
    ["barSize=<durationString>", "(required) The data granularity."],
];
const EXAMPLE_TEXT = "historical-data-updates.js -conid=3691937 -exchange=SMART -barSize=15 mins";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintPositionsApp extends ib_api_next_app_1.IBApiNextApp {
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
        if (!this.cmdLineArgs.barSize) {
            this.error("-barSize argument missing.");
        }
        this.subscription$ = this.api
            .getHistoricalDataUpdates({
            conId: this.cmdLineArgs.conid,
            exchange: this.cmdLineArgs.exchange,
        }, this.cmdLineArgs.barSize, what_to_show_1.WhatToShow.MIDPOINT, 1)
            .subscribe({
            next: (bar) => {
                this.printObject(bar);
            },
            error: (err) => {
                this.error(`getHistoricalDataUpdates failed with '${err.error.message}'`);
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
new PrintPositionsApp().start();
//# sourceMappingURL=historical-data-updates.js.map