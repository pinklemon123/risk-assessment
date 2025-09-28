from typing import Dict, Any, List, Tuple

def validate_row(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    rules = cfg.get("dq_rules", {})
    errors: List[str] = []

    for f in rules.get("required", []):
        if row.get(f) in (None, "", []):
            errors.append(f"required:{f}")

    ranges = rules.get("ranges", {})
    amt_rule = ranges.get("amount")
    if amt_rule:
        val = row.get("amount")
        if val is not None:
            if val < amt_rule.get("min", float("-inf")) or val > amt_rule.get("max", float("inf")):
                errors.append("range:amount")

    return (len(errors) == 0, errors)
