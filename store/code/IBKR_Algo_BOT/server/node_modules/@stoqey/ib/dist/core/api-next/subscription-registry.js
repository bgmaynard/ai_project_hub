"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IBApiNextSubscriptionRegistry = void 0;
const map_1 = require("./map");
const subscription_1 = require("./subscription");
/** The log tag */
const LOG_TAG = "IBApiNextSubscriptionRegistry";
/**
 * @internal
 *
 * An entry on the subscription registry.
 */
class RegistryEntry {
    /**
     * Create a new [[RegistryEntry]] object.
     *
     * @param eventName The [[IBApi]] event name.
     * @param callback The event callback handler.
     */
    constructor(eventName, callback) {
        this.eventName = eventName;
        this.callback = callback;
        /** Map of all active subscriptions, with reqId as key. */
        this.subscriptions = new Map();
        this.listener = (...eventArgs) => {
            this.callback(this.subscriptions, ...eventArgs);
        };
    }
}
/**
 * @internal
 *
 * The subscription registry as used by [[IBApiNext]].
 *
 * The subscription registry maintains the list of all currently
 * registered subscriptions. See [[IBApiNext.register]] about how
 * register a subscription.
 */
class IBApiNextSubscriptionRegistry {
    /**
     * Create an [[IBApiNextSubscriptionRegistry]] instance.
     *
     * @param api The [[IBApiAutoConnection]] instance for event listener registration and
     * invoking TWS API.
     * @param apiNext The [[IBApiNext]] instance for observing the connection state.
     */
    constructor(api, apiNext) {
        this.api = api;
        this.apiNext = apiNext;
        /** A Map containing the subscription registry, with event name as key. */
        this.entries = new map_1.IBApiNextMap();
    }
    /**
     * Register a subscription.
     *
     * @param requestFunction A callback, invoked when the start request shall be send to TWS.
     * @param cancelFunction A callback, invoked when the cancel request shall be send to TWS.
     * @param eventHandler Array of IB API event, callback function to handle this event.
     * @param instanceId When not undefined, this an id that uniquely identifies
     * the subscription instance. This can be used to avoid creation of multiple subscriptions,
     * that will end up on same TWS request (i.e. request same market data multiple times), but an
     * existing subscription instance will be re-used if same instanceId does already exist.
     * As a general rule: don't use instanceId when there is a reqId. Use it everywhere else.
     */
    register(requestFunction, cancelFunction, // eslint-disable-line @typescript-eslint/no-invalid-void-type
    eventHandler, instanceId) {
        // get the existing registry entries, or add if not existing yet
        const entries = [];
        eventHandler.forEach((handler) => {
            const eventName = handler[0];
            const callback = handler[1];
            const entry = this.entries.getOrAdd(eventName, () => {
                const entry = new RegistryEntry(eventName, callback);
                this.apiNext.logger.debug(LOG_TAG, `Add RegistryEntry for EventName.${eventName}`);
                this.api.addListener(eventName, entry.listener);
                return entry;
            });
            entries.push(entry);
        });
        // lookup subscription by instance id
        let subscription;
        if (instanceId) {
            entries.forEach((entry) => {
                const values = entry.subscriptions.values();
                while (!subscription) {
                    const it = values.next();
                    if (it.done) {
                        break;
                    }
                    if (it.value.instanceId === instanceId) {
                        subscription = it.value;
                    }
                }
            });
        }
        // create new subscription
        if (!subscription) {
            subscription = new subscription_1.IBApiNextSubscription(this.apiNext, () => {
                requestFunction(subscription.reqId);
            }, () => {
                if (cancelFunction) {
                    cancelFunction(subscription.reqId);
                }
            }, () => {
                entries.forEach((entry) => {
                    entry.subscriptions.delete(subscription.reqId);
                    if (!entry.subscriptions.size) {
                        this.api.removeListener(entry.eventName, entry.listener);
                        this.apiNext.logger.debug(LOG_TAG, `Remove RegistryEntry for EventName.${entry.eventName}.`);
                        this.entries.delete(entry.eventName);
                    }
                });
                this.apiNext.logger.debug(LOG_TAG, `Deleted IBApiNextSubscription for ${subscription.reqId}.`);
            }, instanceId);
            entries.forEach((entry) => {
                this.apiNext.logger.debug(LOG_TAG, `Add IBApiNextSubscription on EventName.${entry.eventName} for ${subscription.reqId}.`);
                entry.subscriptions.set(subscription.reqId, subscription);
            });
        }
        // create an observable on the subscription
        return subscription.createObservable();
    }
    /**
     * Dispatch an error into the subscription that owns the given request id.
     */
    dispatchError(error) {
        this.entries.forEach((entry) => {
            entry.subscriptions.get(error.reqId)?.error(error);
        });
    }
}
exports.IBApiNextSubscriptionRegistry = IBApiNextSubscriptionRegistry;
//# sourceMappingURL=subscription-registry.js.map