"use strict";
/**
 * This App will print IBKR account place new orders to console.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const ib_api_next_app_1 = require("./common/ib-api-next-app");
/////////////////////////////////////////////////////////////////////////////////
// The help text.                                                              //
/////////////////////////////////////////////////////////////////////////////////
const DESCRIPTION_TEXT = "Get commission report.";
const USAGE_TEXT = "Usage: commission-reports.js <options>";
const OPTION_ARGUMENTS = [];
const EXAMPLE_TEXT = "commission-reports.js  -clientId=0";
//////////////////////////////////////////////////////////////////////////////
// The App code                                                             //
//////////////////////////////////////////////////////////////////////////////
class CommissionReportApp extends ib_api_next_app_1.IBApiNextApp {
    constructor() {
        super(DESCRIPTION_TEXT, USAGE_TEXT, OPTION_ARGUMENTS, EXAMPLE_TEXT);
    }
    /**
     * Start the app.
     */
    start() {
        super.start();
        const executionFilter = {
            clientId: "0",
        };
        this.api.getCommissionReport(executionFilter).then((commissionReports) => {
            this.printObject(commissionReports);
            this.exit();
        }, (error) => {
            this.printObject(error);
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
new CommissionReportApp().start();
//# sourceMappingURL=commission-reports.js.map