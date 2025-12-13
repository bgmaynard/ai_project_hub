"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const event_name_1 = require("../../../api/data/enum/event-name");
describe("RxJS Wrapper: getCommissionReport", () => {
    test("Promise result", (done) => {
        // create IBApiNext
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        const commissionReports = [
            {
                execId: "0000e0d5.619dbad4.01.01",
                commission: 1,
                currency: "USD",
                yieldRedemptionDate: 0,
            },
        ];
        const executionFilter = {};
        const reqId = 1;
        apiNext.getCommissionReport(executionFilter).then((data) => {
            expect(data.length).toEqual(1);
            expect(data[0]).toMatchObject(commissionReports);
            done();
        });
        api.emit(event_name_1.EventName.commissionReport, commissionReports);
        api.emit(event_name_1.EventName.execDetailsEnd, reqId);
    });
});
//# sourceMappingURL=get-commission-report.test.js.map