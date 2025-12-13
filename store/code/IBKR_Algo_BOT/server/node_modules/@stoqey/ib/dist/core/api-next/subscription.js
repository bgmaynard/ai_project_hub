"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextSubscription = void 0;
const rxjs_1 = require("rxjs");
const operators_1 = require("rxjs/operators");
const connection_state_1 = require("../../api-next/common/connection-state");
/**
 * @internal
 *
 * This class implements the management of a subscription on the TWS API.
 *
 * It provides a method to create a [[Observable]] for starting / stopping
 * the TWS subscription and observing changes.
 *
 * The class will take care to call the request and cancel functions when needed.
 */
class IBApiNextSubscription {
    /**
     * Create a [[IBApiNextSubscription]] object.
     *
     * @param api The [[IBApiNext]] instance.
     * @param requestFunction A callback, invoked when the start request shall be send to TWS.
     * @param cancelFunction A callback, invoked when the cancel request shell be send to TWS.
     * @param cleanupFunction A callback, invoked when the last observer has unsubscribed from the subject.
     */
    constructor(api, requestFunction, cancelFunction, cleanupFunction, instanceId) {
        this.api = api;
        this.requestFunction = requestFunction;
        this.cancelFunction = cancelFunction;
        this.cleanupFunction = cleanupFunction;
        this.instanceId = instanceId;
        /** Number of active observers. */
        this.observersCount = 0;
        /** The replay subject, holding the latest emitted values. */
        this.subject = new rxjs_1.ReplaySubject(1);
        /** To prepare RxJS v8 that will remove subject.hasError */
        this.hasError = false;
        /** @internal True when the end-event on an enumeration request has been received, false otherwise. */
        this.endEventReceived = false;
        this.reqId = api.nextReqId;
    }
    /** Get the last 'all' value as send to subscribers. */
    get lastAllValue() {
        return this._lastAllValue;
    }
    /** @internal Set the last 'all' value without publishing it to subscribers. For internal use only. */
    set lastAllValue(value) {
        this._lastAllValue = value;
    }
    /**
     * Send the next value to subscribers.
     *
     * @param value: The next value.
     */
    next(value) {
        this._lastAllValue = value.all;
        this.subject.next(value);
    }
    /** Signal to subscribed that the options is complete. */
    complete() {
        this.subject.complete();
    }
    /**
     * Send an error to subscribers, reset latest value to
     * undefined and cancel TWS subscription.
     *
     * @param error: The [[IBApiError]] object.
     */
    error(error) {
        delete this._lastAllValue;
        // this.endEventReceived = false;
        this.hasError = true;
        this.subject.error(error);
        this.cancelTwsSubscription();
    }
    /**
     * Create an Observable to start/stop the subscription on
     * TWS, receive update and error events.
     */
    createObservable() {
        return new rxjs_1.Observable((subscriber) => {
            // create new subject and reqId if there is an has error
            if (this.hasError) {
                this.hasError = false;
                this.subject = new rxjs_1.ReplaySubject(1);
                this.reqId = this.api.nextReqId;
            }
            // subscribe on subject
            const subscription$ = this.subject
                .pipe((0, operators_1.map)((val, index) => {
                return index === 0
                    ? {
                        all: val.all,
                        added: val.all,
                    }
                    : val;
            }))
                .subscribe(subscriber);
            // request from TWS if first subscriber
            if (this.observersCount++ === 0) {
                this.requestTwsSubscription();
            }
            // this.observersCount++; moved into "if" condition above
            // handle unsubscribe
            return () => {
                subscription$.unsubscribe();
                // this.observersCount--; moved into "if" condition below
                if (--this.observersCount <= 0) {
                    this.cancelTwsSubscription();
                    this.cleanupFunction();
                }
            };
        });
    }
    /**
     * Invoke TWS request function and setup connection state subscription
     */
    requestTwsSubscription() {
        // subscribe on connection state: send TWS request when 'connected' state is signaled
        if (!this.connectionState$) {
            this.connectionState$ = this.api.connectionState.subscribe((state) => {
                if (state === connection_state_1.ConnectionState.Connected) {
                    delete this._lastAllValue;
                    this.endEventReceived = false;
                    this.requestFunction();
                }
            });
        }
    }
    /**
     * Invoke TWS cancel function and unsubscribe from connection state
     */
    cancelTwsSubscription() {
        this.connectionState$?.unsubscribe();
        delete this.connectionState$;
        if (this.api.isConnected) {
            this.cancelFunction();
        }
    }
}
exports.IBApiNextSubscription = IBApiNextSubscription;
//# sourceMappingURL=subscription.js.map