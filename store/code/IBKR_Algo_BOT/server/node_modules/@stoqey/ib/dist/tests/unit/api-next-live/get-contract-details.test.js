"use strict";
/**
 * This file implements tests for the [[IBApiNext.getContractDetails]] function.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const __1 = require("../../..");
const logger_1 = __importDefault(require("../../../common/logger"));
const contracts_1 = require("../sample-data/contracts");
describe("ApiNext: getContractDetails()", () => {
    jest.setTimeout(5_000);
    const clientId = Math.floor(Math.random() * 32766) + 1; // ensure unique client
    let api;
    let error$;
    beforeEach(() => {
        api = new __1.IBApiNext();
        if (!error$) {
            error$ = api.errorSubject.subscribe((error) => {
                if (error.reqId === -1) {
                    logger_1.default.warn(`${error.error.message} (Error #${error.code})`);
                }
                else {
                    logger_1.default.error(`${error.error.message} (Error #${error.code}) ${error.advancedOrderReject ? error.advancedOrderReject : ""}`);
                }
            });
        }
        try {
            api.connect(clientId);
        }
        catch (error) {
            logger_1.default.error(error.message);
        }
    });
    afterEach(() => {
        if (api) {
            api.disconnect();
            api = undefined;
        }
    });
    test("Stock contract details", (done) => {
        const ref_contract = contracts_1.sample_stock;
        api
            .getContractDetails(ref_contract)
            .then((result) => {
            expect(result.length).toBeGreaterThan(0);
            expect(result[0].contract.symbol).toEqual(ref_contract.symbol);
            expect(result[0].contract.secType).toEqual(ref_contract.secType);
            done();
        })
            .catch((err) => {
            done(`getContractDetails failed with '${err.error.message}' (Error #${err.code})`);
        });
    });
    test("Future contract details", (done) => {
        const ref_contract = contracts_1.sample_future;
        api
            .getContractDetails(ref_contract)
            .then((result) => {
            expect(result.length).toBeGreaterThan(0);
            expect(result[0].contract.symbol).toEqual(ref_contract.symbol);
            expect(result[0].contract.secType).toEqual(ref_contract.secType);
            done();
        })
            .catch((err) => {
            done(`getContractDetails failed with '${err.error.message}' (Error #${err.code})`);
        });
    });
    test("Crypto contract details", (done) => {
        const ref_contract = contracts_1.sample_crypto;
        api
            .getContractDetails(ref_contract)
            .then((result) => {
            expect(result.length).toBeGreaterThan(0);
            expect(result[0].contract.symbol).toEqual(ref_contract.symbol);
            expect(result[0].contract.secType).toEqual(ref_contract.secType);
            done();
        })
            .catch((err) => {
            done(`getContractDetails failed with '${err.error.message}' (Error #${err.code})`);
        });
    });
    test("Option contract details", (done) => {
        const ref_contract = contracts_1.sample_option;
        api
            .getContractDetails(ref_contract)
            .then((result) => {
            expect(result.length).toBeGreaterThan(0);
            expect(result[0].contract.symbol).toEqual(ref_contract.symbol);
            expect(result[0].contract.secType).toEqual(ref_contract.secType);
            done();
        })
            .catch((err) => {
            done(`getContractDetails failed with '${err.error.message}' (Error #${err.code})`);
        });
    });
    test("Bond contract details", (done) => {
        const ref_contract = contracts_1.sample_bond;
        api
            .getContractDetails(ref_contract)
            .then((result) => {
            expect(result.length).toBeGreaterThan(0);
            // expect(result[0].contract.symbol).toEqual(ref_contract.symbol);
            expect(result[0].contract.secType).toEqual(ref_contract.secType);
            done();
        })
            .catch((err) => {
            done(`getContractDetails failed with '${err.error.message}' (Error #${err.code})`);
        });
    });
});
//# sourceMappingURL=get-contract-details.test.js.map