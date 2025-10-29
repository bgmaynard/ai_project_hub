import { Observable } from "rxjs";
import { IBApiNext, IBApiNextError, ItemListUpdate } from "../../api-next";
import { IBApiNextItemListUpdate } from "./item-list-update";
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
export declare class IBApiNextSubscription<T> {
    private api;
    private requestFunction;
    private cancelFunction;
    private cleanupFunction;
    readonly instanceId?: string;
    /**
     * Create a [[IBApiNextSubscription]] object.
     *
     * @param api The [[IBApiNext]] instance.
     * @param requestFunction A callback, invoked when the start request shall be send to TWS.
     * @param cancelFunction A callback, invoked when the cancel request shell be send to TWS.
     * @param cleanupFunction A callback, invoked when the last observer has unsubscribed from the subject.
     */
    constructor(api: IBApiNext, requestFunction: () => void, cancelFunction: () => void, cleanupFunction: () => void, instanceId?: string);
    /** The request id of this subscription. */
    reqId: number;
    /** Number of active observers. */
    private observersCount;
    /** The replay subject, holding the latest emitted values. */
    private subject;
    /** To prepare RxJS v8 that will remove subject.hasError */
    private hasError;
    /** The last 'all' value as send to subscribers. */
    private _lastAllValue?;
    /** The [[Subscription]] on the connection state. */
    private connectionState$?;
    /** @internal True when the end-event on an enumeration request has been received, false otherwise. */
    endEventReceived: boolean;
    /** Get the last 'all' value as send to subscribers. */
    get lastAllValue(): T | undefined;
    /** @internal Set the last 'all' value without publishing it to subscribers. For internal use only. */
    set lastAllValue(value: T);
    /**
     * Send the next value to subscribers.
     *
     * @param value: The next value.
     */
    next(value: IBApiNextItemListUpdate<T>): void;
    /** Signal to subscribed that the options is complete. */
    complete(): void;
    /**
     * Send an error to subscribers, reset latest value to
     * undefined and cancel TWS subscription.
     *
     * @param error: The [[IBApiError]] object.
     */
    error(error: IBApiNextError): void;
    /**
     * Create an Observable to start/stop the subscription on
     * TWS, receive update and error events.
     */
    createObservable(): Observable<ItemListUpdate<T>>;
    /**
     * Invoke TWS request function and setup connection state subscription
     */
    private requestTwsSubscription;
    /**
     * Invoke TWS cancel function and unsubscribe from connection state
     */
    private cancelTwsSubscription;
}
