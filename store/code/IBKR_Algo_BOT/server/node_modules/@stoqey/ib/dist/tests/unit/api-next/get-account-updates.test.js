"use strict";
/**
 * This file implements tests for the [[IBApiNext.getAccountSummary]] function.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const contracts_1 = require("../sample-data/contracts");
describe("RxJS Wrapper: getAccountUpdates()", () => {
    test("Update multicast", (done) => {
        const apiNext = new __1.IBApiNext();
        const api = apiNext.api;
        // testing values
        const accountId1 = "DU123456";
        const accountId2 = "DU654321";
        const currency = "USD";
        apiNext
            .getAccountUpdates()
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (update) => {
                if (update.changed?.timestamp == "now") {
                    expect(update.all.value.get(accountId1)).toBeDefined();
                    expect(update.all.value.get(accountId1).get("tag").get(currency).value).toBe("value1");
                    expect(update.all.portfolio.get(accountId1)).toBeDefined();
                    expect(update.all.portfolio.get(accountId1)[0].account).toBe(accountId1);
                    expect(update.all.portfolio.get(accountId1)[0].contract.symbol).toBe(contracts_1.sample_etf.symbol);
                    expect(update.all.value.get(accountId2)).toBeDefined();
                    expect(update.all.value.get(accountId2).get("tag").get(currency).value).toBe("value2");
                    expect(update.all.portfolio.get(accountId2)).toBeDefined();
                    expect(update.all.portfolio.get(accountId2)[0].account).toBe(accountId2);
                    expect(update.all.portfolio.get(accountId2)[0].contract.symbol).toBe(contracts_1.sample_stock.symbol);
                }
                else if (update.changed?.timestamp == "later")
                    done();
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        apiNext
            .getAccountUpdates(accountId1)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (update) => {
                if (update.changed?.timestamp == "now") {
                    expect(update.all.value.get(accountId1)).toBeDefined();
                    expect(update.all.value.get(accountId1).get("tag").get(currency).value).toBe("value1");
                    expect(update.all.portfolio.get(accountId1)).toBeDefined();
                    expect(update.all.portfolio.get(accountId1)[0].account).toBe(accountId1);
                    expect(update.all.portfolio.get(accountId1)[0].contract.symbol).toBe(contracts_1.sample_etf.symbol);
                    expect(update.all.value.get(accountId2)).toBeUndefined();
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        apiNext
            .getAccountUpdates(accountId2)
            // eslint-disable-next-line rxjs/no-ignored-subscription
            .subscribe({
            next: (update) => {
                if (update.changed?.timestamp == "now") {
                    expect(update.all.value.get(accountId1)).toBeUndefined();
                    expect(update.all.value.get(accountId2)).toBeDefined();
                    expect(update.all.value.get(accountId2).get("tag").get(currency).value).toBe("value2");
                    expect(update.all.portfolio.get(accountId2)).toBeDefined();
                    expect(update.all.portfolio.get(accountId2)[0].account).toBe(accountId2);
                    expect(update.all.portfolio.get(accountId2)[0].contract.symbol).toBe(contracts_1.sample_stock.symbol);
                }
            },
            error: (error) => {
                fail(error.error.message);
            },
        });
        api.emit(__1.EventName.updateAccountValue, "tag", "value1", currency, accountId1);
        api.emit(__1.EventName.updatePortfolio, contracts_1.sample_etf, 1, 10, 100, 9, 10, 0, accountId1);
        api.emit(__1.EventName.accountDownloadEnd, accountId1);
        api.emit(__1.EventName.updateAccountValue, "tag", "value2", currency, accountId2);
        api.emit(__1.EventName.updatePortfolio, contracts_1.sample_stock, 1, 10, 100, 9, 10, 0, accountId2);
        api.emit(__1.EventName.accountDownloadEnd, accountId2);
        api.emit(__1.EventName.updateAccountTime, "now");
        api.emit(__1.EventName.updateAccountTime, "later");
    });
});
//# sourceMappingURL=get-account-updates.test.js.map