"""Event-by-event Data Siege defense using only the documented toolkit API."""
import math
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _invalid(pillar, why="insufficient or invalid signal data"):
    return Verdict(alert=False, pillar=pillar, reason=why)


def _number(value):
    # bool is an int subclass, but is not a meaningful metric here.
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _rpc(call, pillar):
    """Run one metered request and fail closed on transport/tool errors."""
    try:
        result = call()
    except Exception:
        return None
    if not isinstance(result, dict) or "error" in result:
        return None
    return result


def _baseline_number(ctx, name):
    value = ctx.baseline.get(name)
    return value if _number(value) else None


def check_data_batch(payload, ctx):
    pillar = "checks"
    if not isinstance(payload, dict) or not payload.get("batch_id"):
        return _invalid(pillar, "missing batch_id")
    result = _rpc(lambda: ctx.tools.batch_profile(payload["batch_id"]), pillar)
    if result is None:
        return _invalid(pillar, "batch profile unavailable")
    nulls = result.get("null_rate")
    if not isinstance(nulls, dict):
        return _invalid(pillar)
    values = [result.get("row_count"), nulls.get("customer_id"),
              result.get("mean_amount"), result.get("staleness_min")]
    bounds = [_baseline_number(ctx, "row_count_min"),
              _baseline_number(ctx, "row_count_max"),
              _baseline_number(ctx, "null_rate_max"),
              _baseline_number(ctx, "mean_amount_min"),
              _baseline_number(ctx, "mean_amount_max"),
              _baseline_number(ctx, "staleness_min_max")]
    if not all(_number(v) for v in values) or any(v is None for v in bounds):
        return _invalid(pillar)
    row_count, null_rate, mean_amount, staleness = values
    row_min, row_max, null_max, amount_min, amount_max, stale_max = bounds
    issues = []
    if row_count < row_min or row_count > row_max:
        issues.append("row_count")
    if null_rate > null_max:
        issues.append("customer_id_null_rate")
    if mean_amount < amount_min or mean_amount > amount_max:
        issues.append("mean_amount")
    if staleness > stale_max:
        issues.append("staleness")
    return Verdict(alert=bool(issues), pillar=pillar,
                   reason=("out of bounds: " + ", ".join(issues)) if issues else "within baseline")


def check_contract_checkpoint(payload, ctx):
    pillar = "contracts"
    if (not isinstance(payload, dict) or not payload.get("contract_id")
            or not payload.get("checkpoint_batch_id")):
        return _invalid(pillar, "missing contract reference")
    result = _rpc(lambda: ctx.tools.contract_diff(
        payload["contract_id"], payload["checkpoint_batch_id"]), pillar)
    if result is None:
        return _invalid(pillar, "contract diff unavailable")
    delay = result.get("freshness_delay_min")
    violations = result.get("violations")
    delay_max = _baseline_number(ctx, "freshness_delay_max_min")
    if (not _number(delay) or delay_max is None or not isinstance(violations, list)
            or any(not isinstance(item, str) for item in violations)):
        return _invalid(pillar)
    issues = list(violations)
    if delay > delay_max:
        issues.append("freshness_delay")
    return Verdict(alert=bool(issues), pillar=pillar,
                   reason=("contract issues: " + ", ".join(issues)) if issues else "contract within limits")


def check_lineage_run(payload, ctx):
    pillar = "lineage"
    if not isinstance(payload, dict) or not payload.get("run_id"):
        return _invalid(pillar, "missing run_id")
    result = _rpc(lambda: ctx.tools.lineage_graph_slice(payload["run_id"]), pillar)
    if result is None:
        return _invalid(pillar, "lineage slice unavailable")
    duration = result.get("duration_ms")
    duration_max = _baseline_number(ctx, "lineage_duration_ms_max")
    if not _number(duration) or duration_max is None:
        return _invalid(pillar)
    issues = []
    if duration > duration_max:
        issues.append("duration")

    # Compare graph shape only when this event explicitly supplies its expected
    # shape; the defense does not assume a topology or memorize run IDs.
    if "expected_upstream" in payload:
        actual = result.get("actual_upstream")
        expected = payload.get("expected_upstream")
        if (not isinstance(actual, list) or not isinstance(expected, list)
                or any(not isinstance(item, str) for item in actual + expected)):
            return _invalid(pillar)
        if sorted(actual) != sorted(expected):
            issues.append("upstream_mismatch")
    if "expected_downstream_count" in payload:
        actual = result.get("actual_downstream_count")
        expected = payload.get("expected_downstream_count")
        if not _number(actual) or not _number(expected):
            return _invalid(pillar)
        if actual != expected:
            issues.append("downstream_count_mismatch")
    return Verdict(alert=bool(issues), pillar=pillar,
                   reason=("lineage anomaly: " + ", ".join(issues)) if issues else "within lineage limits")


def check_feature_materialization(payload, ctx):
    pillar = "ai_infra"
    if (not isinstance(payload, dict) or not payload.get("feature_view")
            or not payload.get("batch_id")):
        return _invalid(pillar, "missing feature reference")
    result = _rpc(lambda: ctx.tools.feature_drift(
        payload["feature_view"], payload["batch_id"]), pillar)
    if result is None:
        return _invalid(pillar, "feature drift unavailable")
    shift = result.get("mean_shift_sigma")
    shift_max = _baseline_number(ctx, "feature_mean_shift_sigma_max")
    if not _number(shift) or shift_max is None:
        return _invalid(pillar)
    alert = shift > shift_max
    return Verdict(alert=alert, pillar=pillar,
                   reason="feature mean shift exceeds baseline" if alert else "within feature baseline")


def check_embedding_batch(payload, ctx):
    pillar = "ai_infra"
    if (not isinstance(payload, dict) or not payload.get("corpus")
            or not payload.get("chunk_batch_id")):
        return _invalid(pillar, "missing corpus reference")
    result = _rpc(lambda: ctx.tools.embedding_drift(
        payload["corpus"], payload["chunk_batch_id"]), pillar)
    if result is None:
        return _invalid(pillar, "embedding drift unavailable")
    shift = result.get("centroid_shift")
    age = result.get("avg_doc_age_days")
    shift_max = _baseline_number(ctx, "embedding_centroid_shift_max")
    age_max = _baseline_number(ctx, "corpus_avg_doc_age_days_max")
    if not _number(shift) or not _number(age) or shift_max is None or age_max is None:
        return _invalid(pillar)
    issues = []
    if shift > shift_max:
        issues.append("centroid_shift")
    if age > age_max:
        issues.append("document_age")
    return Verdict(alert=bool(issues), pillar=pillar,
                   reason=("embedding issues: " + ", ".join(issues)) if issues else "within embedding limits")
