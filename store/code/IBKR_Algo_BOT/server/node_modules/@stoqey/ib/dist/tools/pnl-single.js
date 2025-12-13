"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text                                                               //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Print real time updates for daily PnL of individual positions..";
const USAGE_TEXT = "Usage: pnl-single.js <options>";
const OPTION_ARGUMENTS = [
    ["account=<account_id>", "(required) Account in which position exists."],
    [
        "conid=<number>",
        "(required) Contract ID (conId) of contract to receive daily PnL updates for.",
    ],
    ["model=<code>", "Model in which position exists."],
];
const EXAMPLE_TEXT = "pnl-single.js -account=DU1234567 -conid=1234567 -watch";
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
        if (!this.cmdLineArgs.account) {
            this.error("-account argument missing.");
        }
        if (!this.cmdLineArgs.conid) {
            this.error("-conid argument missing.");
        }
        this.subscription$ = this.api
            .getPnLSingle(this.cmdLineArgs.account, this.cmdLineArgs.model, this.cmdLineArgs.conid)
            .subscribe({
            next: (pnlSingle) => {
                this.printObject(pnlSingle);
                if (!this.cmdLineArgs.watch) {
                    this.stop();
                }
            },
            error: (err) => {
                this.error(`getPnLSingle failed with '${err.error.message}'`);
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
//# sourceMappingURL=pnl-single.js.map