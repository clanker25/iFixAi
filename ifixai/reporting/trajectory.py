"""Self-contained private evidence reports for recorded trajectory assurance."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from ifixai.trajectory.models import AuditBundle


def policy_comparison(original: AuditBundle, scenario: AuditBundle) -> dict[str, Any]:
    def keys(bundle: AuditBundle) -> set[tuple[str, str, str]]:
        return {(item.rule, item.code, item.event_id) for item in bundle.report.violations}

    before, after = keys(original), keys(scenario)
    return {"original_integrity_digest": original.integrity_digest,
            "original_policy_digest": original.report.policy_digest,
            "scenario_policy_digest": scenario.report.policy_digest,
            "policy_changed": original.report.policy_digest != scenario.report.policy_digest,
            "original_decision": original.report.decision, "scenario_decision": scenario.report.decision,
            "resolved": sorted(before - after), "new": sorted(after - before),
            "persistent": sorted(before & after),
            "limitation": "Re-evaluates retained facts under an alternate policy; does not re-execute or prove an intervention."}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def render_trajectory_html(bundle: AuditBundle, comparison: dict[str, Any] | None = None) -> str:
    escape = html.escape
    report = bundle.report
    anchors = {event.event_id: f"event-{index}" for index, event in enumerate(bundle.capture.events)}

    def link(event_id: str) -> str:
        label = escape(event_id)
        if event_id not in anchors:
            return f"<code>{label} (missing)</code>"
        return f'<a href="#{anchors[event_id]}">{label}</a>'

    rows = []
    for event in bundle.capture.events:
        payload = event.payload.model_dump(mode="json")
        details = " · ".join(f"{key}: {value}" for key, value in payload.items() if key not in {"kind", "arguments_digest", "content_digest", "citations"})
        dependencies = " ← ".join(link(parent) for parent in event.parents) or "root"
        rows.append(f'<tr id="{anchors[event.event_id]}"><td>{event.sequence}</td><td>{link(event.event_id)}<br><small>{event.time_ns} ns</small></td><td><b>{escape(payload["kind"])}</b><br>{escape(details)}</td><td>{dependencies}</td></tr>')
    findings = []
    for item in report.violations:
        witnesses = " → ".join(link(key) for key in item.witness_ids)
        findings.append(f'<li><b>{escape(item.code)}</b> · {link(item.event_id)}<p>{escape(item.message)}</p><div>Witness: {witnesses or "none"}</div></li>')
    coverage = []
    for gap in report.coverage_issues:
        coverage.append(f'<li><b>{escape(gap.code)}</b> {link(gap.event_id) if gap.event_id else ""}<p>{escape(gap.message)}</p></li>')
    families = "".join(f'<tr><td>{escape(family)}</td><td>{escape(status)}</td><td>{report.family_counts.get(family, 0)}</td></tr>' for family, status in report.family_status.items())
    budgets = []
    for rule in bundle.policy.budgets:
        key = "trace" if rule.scope == "trace" else f"actor:{rule.actor_id}"
        used = report.budget_totals.get(key, 0)
        width = min(100, used * 100 // rule.limit_units) if rule.limit_units else (100 if used else 0)
        budgets.append(f'<li>{escape(key)}: <b>{used}</b> / {rule.limit_units} declared units<div class="bar"><span style="width:{width}%"></span></div></li>')
    scenario = f'<section><h2>Alternate policy scenario</h2><pre>{escape(_json(comparison))}</pre></section>' if comparison else ""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>iFixAi · Recorded trajectory assurance</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0b1220;color:#e6edf7;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:48px 24px}}h1{{font-size:42px;line-height:1.15;margin:12px 0}}h2{{margin-top:0}}.eyebrow,small{{color:#9eacc4}}section{{background:#111d30;border:1px solid #293b54;padding:24px;border-radius:14px;margin:20px 0}}.decision{{display:inline-block;background:#284864;padding:6px 14px;border-radius:6px;font-weight:bold}}a{{color:#9bd1ff}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #293b54;vertical-align:top;overflow-wrap:anywhere}}li{{margin:12px 0}}p{{margin:8px 0}}code,pre{{font:13px/1.5 ui-monospace,monospace;overflow-wrap:anywhere;white-space:pre-wrap}}input{{background:#0b1220;border:1px solid #607895;color:#fff;padding:12px;width:100%;margin-bottom:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px}}.bar{{height:8px;background:#293b54;border-radius:4px;overflow:hidden}}.bar span{{height:100%;background:#55bdcf;display:block}}tr:target{{background:#233e5b}}.table-wrap{{overflow-x:auto}}
</style></head><body><main><div class="eyebrow">IFIXAI / OFFLINE EVIDENCE</div><h1>Recorded trajectory assurance</h1><p class="decision">{escape(report.decision)}</p><p>Policy: {escape(bundle.policy.name)} · {len(bundle.capture.events)} retained events · {report.completed_actions} completed actions</p>
<p>This is a passive audit of collector-declared facts. CLEAR applies to the evaluated policy and retained evidence. It is not runtime prevention, authenticated capture, semantic truth or deployment certification.</p>
<section class="grid"><div><b>{len(report.violations)} observed violations</b><p>Linked witnesses identify the recorded events behind each finding.</p></div><div><b>{len(report.coverage_issues)} evidence gaps</b><p>Missing records or unevaluated families limit the conclusion.</p></div><div><b>{report.failed_actions} failed actions</b><p>Failure does not establish that side effects were absent.</p></div></section>
<section><h2>Findings and witnesses</h2><ul>{''.join(findings) or '<li>No observed violations.</li>'}</ul></section>
<section><h2>Evidence coverage</h2><table><thead><tr><th>Family</th><th>Status</th><th>Checked events</th></tr></thead><tbody>{families}</tbody></table><ul>{''.join(coverage) or '<li>No recorded coverage gaps.</li>'}</ul></section>
<section><h2>Shared budget ledger</h2><ul>{''.join(budgets) or '<li>No budget policy configured.</li>'}</ul><p>Integer units are supplied by the application; this is not a billing statement.</p></section>
{scenario}<section><h2>Recorded dependency graph and timeline</h2><p>Parents are explicit recorded links. Ordering alone does not confer authority. Select a witness to jump to its event.</p><input id="search" aria-label="Search retained events" placeholder="Filter by actor, event or action"><div class="table-wrap"><table><thead><tr><th>Sequence</th><th>Event / time</th><th>Recorded fact</th><th>Parents ←</th></tr></thead><tbody id="events">{''.join(rows)}</tbody></table></div></section>
<section><h2>Replay identity and limits</h2><p>Trace: <code>{escape(bundle.capture.trace_id)}</code> · Collector: {escape(bundle.capture.collector_id)} · Format: {escape(bundle.capture.source_format)}</p><p>Bundle SHA-256: <code>{bundle.integrity_digest}</code></p><ul>{''.join('<li>'+escape(limit)+'</li>' for limit in report.limitations)}</ul><p>A checksum detects changes to the bundle; it does not establish the truth of the recorded events.</p></section></main>
<script>const search=document.getElementById('search');const rows=Array.from(document.querySelectorAll('#events tr'));search.addEventListener('input',()=>{{const q=search.value.toLowerCase();rows.forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q));}});window.addEventListener('hashchange',()=>{{const target=document.getElementById(location.hash.slice(1));if(target){{target.hidden=false;target.scrollIntoView();}}}});</script></body></html>'''


def render_trajectory_markdown(bundle: AuditBundle, comparison: dict[str, Any] | None = None) -> str:
    # Escaping untrusted labels prevents generated Markdown from embedding links/images.
    def safe(text: str) -> str:
        for character in ("\\", "`", "*", "_", "[", "]", "<", ">", "|", "#"):
            text = text.replace(character, "\\" + character)
        return text.replace("\n", " ").replace("\r", " ")

    lines = ["# Recorded trajectory assurance", "", f"Decision: **{bundle.report.decision}**", "",
             f"Policy: {safe(bundle.policy.name)}. Retained events: {len(bundle.capture.events)}.", "",
             "Passive audit of collector-declared facts; not runtime prevention or certification.", "", "## Findings", ""]
    lines.extend(f"- {safe(item.code)} at {safe(item.event_id)}: {safe(item.message)}. Witnesses: {', '.join(safe(w) for w in item.witness_ids)}" for item in bundle.report.violations)
    if not bundle.report.violations:
        lines.append("No observed violations.")
    lines.extend(["", "## Coverage", ""])
    lines.extend(f"- {safe(item.code)}: {safe(item.message)}" for item in bundle.report.coverage_issues)
    lines.extend(f"- {safe(family)}: {status}" for family, status in bundle.report.family_status.items())
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {safe(limit)}" for limit in bundle.report.limitations)
    if comparison:
        lines.extend(["", "## Alternate policy scenario", "", safe(_json(comparison))])
    lines.extend(["", f"Bundle SHA-256: `{bundle.integrity_digest}`", ""])
    return "\n".join(lines)


def write_trajectory_reports(bundle: AuditBundle, output: str | Path, *, comparison: dict[str, Any] | None = None) -> Path:
    """Create an owner-only NEW directory; refuse every existing destination."""
    destination = Path(output)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    files = {"bundle.json": _json(bundle.model_dump(mode="json")),
             "report.json": _json(bundle.report.model_dump(mode="json")),
             "report.html": render_trajectory_html(bundle, comparison),
             "report.md": render_trajectory_markdown(bundle, comparison)}
    if comparison:
        files["comparison.json"] = _json(comparison)
    for name, content in files.items():
        descriptor = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
    return destination
