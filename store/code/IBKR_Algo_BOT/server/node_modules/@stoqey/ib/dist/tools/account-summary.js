"use strict";
/**
 * This App will print IBKR account summaries to console.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// Default options values.                                                     //
/////////////////////////////////////////////////////////////////////////////////
/** Default group if no -group argument is on command line. */
const DEFAULT_GROUP = "All";
/** Default tags if no -tags argument is on command line. */
const DEFAULT_TAGS = "NetLiquidation,TotalCashValue,GrossPositionValue";
/////////////////////////////////////////////////////////////////////////////////
// The help text.                                                              //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Prints the account summaries.";
const USAGE_TEXT = "Usage: account-summary.js <options>";
const OPTION_ARGUMENTS = [
    [
        "group=<name>",
        `Advisor Account Group name. Default is '${DEFAULT_GROUP}'.`,
    ],
    [
        "tags=<tag list>",
        `A comma separated list with the desired tags. Default is '${DEFAULT_TAGS}'`,
    ],
    [
        "watch",
        "Watch for changes. If specified, the app will keep running and print account summary updates to console as received from TWS. " +
            "If not specified, the app will print a one-time snapshot and than exit.",
    ],
];
const EXAMPLE_TEXT = "account-summary.js -group=All -tags=NetLiquidation,MaintMarginReq -watch";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class PrintAccountSummaryApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        this.subscription$ = this.api
            .getAccountSummary(this.cmdLineArgs.group ?? DEFAULT_GROUP, this.cmdLineArgs.tags ?? DEFAULT_TAGS)
            .subscribe({
            next: (summaries) => {
                this.printObject(summaries);
                if (!this.cmdLineArgs.watch) {
                    this.stop();
                }
            },
            error: (err) => {
                this.error(`getAccountSummary failed with '${err.error.message}'`);
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
new PrintAccountSummaryApp().start();
//# sourceMappingURL=account-summary.js.map