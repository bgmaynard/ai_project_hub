import { CurrencyCode, AccountId, AccountSummaries, AccountSummaryValue, AccountSummaryValues, AccountSummaryTagName, AccountSummaryTagValues, AccountSummariesUpdate } from "../../../../api-next";
import { IBApiNextMap } from "../../map";
import { IBApiNextItemListUpdate } from "../../item-list-update";
/** Mutable version of [[AccountSummaryValues]] */
export declare class MutableAccountSummaryValues extends IBApiNextMap<CurrencyCode, AccountSummaryValue> implements AccountSummaryValues {
}
/** Mutable version of [[AccountSummaryTagValues]] */
export declare class MutableAccountSummaryTagValues extends IBApiNextMap<AccountSummaryTagName, MutableAccountSummaryValues> implements AccountSummaryTagValues {
}
/** Mutable version of [[AccountSummary]] */
export declare class MutableAccountSummaries extends IBApiNextMap<AccountId, MutableAccountSummaryTagValues> implements AccountSummaries {
}
/** Mutable version of [[AccountSummariesUpdate]] */
export declare class MutableAccountSummariesUpdate extends IBApiNextItemListUpdate<MutableAccountSummaries> implements AccountSummariesUpdate {
}
