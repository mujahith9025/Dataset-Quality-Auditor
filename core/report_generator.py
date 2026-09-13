"""
Multi-Format Report Generator
Generates standalone interactive HTML reports, runnable Python data cleaning pipelines, and JSON audit logs.
"""

from typing import Any, Dict, List, Optional
import json
import html
from models.schemas import AuditReport, SeverityEnum, DimensionEnum


class ReportGenerator:
    """Exports dataset audit reports to interactive HTML, runnable Python scripts, and JSON."""

    @staticmethod
    def generate_json_report(report: AuditReport) -> str:
        """Serializes AuditReport to pretty JSON string."""
        return report.model_dump_json(indent=2)

    @staticmethod
    def generate_python_script(report: AuditReport) -> str:
        """Generates a standalone, production-ready Python cleaning pipeline script."""
        code_lines = [
            '"""',
            f'Auto-Generated Data Cleaning Pipeline for: {report.dataset_name}',
            f'Original Data Quality Score: {report.overall_score}/100 (Grade: {report.grade})',
            f'Generated on: {report.created_at}',
            '"""',
            '',
            'import pandas as pd',
            'import numpy as np',
            '',
            'def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:',
            '    """Applies verified automated remediation steps."""',
            '    cleaned = df.copy()',
            ''
        ]

        # Deduplicate actions
        executed_actions = set()
        for rec in report.recommended_actions:
            if rec.get("auto_fixable") and rec.get("suggested_code_snippet"):
                snippet = rec["suggested_code_snippet"]
                if snippet not in executed_actions:
                    executed_actions.add(snippet)
                    code_lines.append(f"    # Step: {rec['title']}")
                    for line in snippet.splitlines():
                        code_lines.append(f"    {line}")
                    code_lines.append("")

        code_lines.extend([
            '    return cleaned',
            '',
            'if __name__ == "__main__":',
            '    # Example usage:',
            '    # df = pd.read_csv("your_data.csv")',
            '    # df_cleaned = clean_dataset(df)',
            '    # df_cleaned.to_csv("cleaned_data.csv", index=False)',
            '    pass',
            ''
        ])

        return "\n".join(code_lines)

    @staticmethod
    def generate_html_report(report: AuditReport, cleaned_report: Optional[AuditReport] = None) -> str:
        """Generates a modern, standalone HTML report with responsive styling."""
        
        # Color palettes based on score
        score = report.overall_score
        score_color = "#10B981" if score >= 85 else ("#F59E0B" if score >= 70 else "#EF4444")
        
        # Build alert cards HTML (sanitized)
        alerts_html = "".join([
            f'<div class="alert-item"><span class="alert-icon">⚠</span> <span>{html.escape(alert)}</span></div>'
            for alert in report.summary_alerts
        ])

        # Build dimension progress bars
        dimensions_html = ""
        for dim_name, ds in report.dimension_scores.items():
            dim_color = "#10B981" if ds.score >= 85 else ("#F59E0B" if ds.score >= 70 else "#EF4444")
            dimensions_html += f"""
            <div class="dim-card">
                <div class="dim-header">
                    <span class="dim-name">{html.escape(dim_name)}</span>
                    <span class="dim-score" style="color: {dim_color};">{ds.score:.1f}/100 ({ds.grade})</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {ds.score}%; background: {dim_color};"></div>
                </div>
                <div class="dim-footer">Weight: {int(ds.weight*100)}%</div>
            </div>
            """

        # Build recommendation steps HTML
        recs_html = ""
        for rec in report.recommended_actions:
            sev_badge_class = f"badge-{rec['severity'].lower()}"
            code_block = f"<pre><code>{html.escape(rec['code_snippet'])}</code></pre>" if rec.get('code_snippet') else ""
            recs_html += f"""
            <div class="rec-card">
                <div class="rec-header">
                    <span class="rec-step">Step {rec['step']}</span>
                    <span class="badge {sev_badge_class}">{html.escape(rec['severity'])}</span>
                    <span class="rec-title">{html.escape(rec['title'])}</span>
                </div>
                <div class="rec-body">
                    <p class="rec-action">{html.escape(rec['action'])}</p>
                    {code_block}
                </div>
            </div>
            """

        # Build column profile table rows
        col_rows_html = ""
        for col_name, prof in report.column_profiles.items():
            miss_color = "#EF4444" if prof.missing_percentage > 20 else ("#F59E0B" if prof.missing_percentage > 0 else "#10B981")
            issues_badge = f'<span class="badge badge-high">{prof.issues_count} issues</span>' if prof.issues_count > 0 else '<span class="badge badge-low">Clean</span>'
            samples_str = ", ".join(str(s) for s in prof.sample_values[:3])
            col_rows_html += f"""
            <tr>
                <td><strong>{html.escape(col_name)}</strong> {'<span class="target-tag">TARGET</span>' if prof.is_target else ''}</td>
                <td><span class="type-tag">{html.escape(prof.inferred_type)}</span></td>
                <td style="color: {miss_color};">{prof.missing_count} ({prof.missing_percentage}%)</td>
                <td>{prof.unique_count} ({prof.unique_percentage}%)</td>
                <td class="sample-cell" title="{html.escape(samples_str)}">{html.escape(samples_str[:40])}...</td>
                <td>{issues_badge}</td>
            </tr>
            """

        escaped_ds_name = html.escape(report.dataset_name)
        # HTML Template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Quality Audit Report - {escaped_ds_name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0F172A;
            --surface: #1E293B;
            --surface-hover: #334155;
            --border: #334155;
            --text: #F8FAFC;
            --text-muted: #94A3B8;
            --primary: #3B82F6;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 2rem;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .title {{ font-size: 1.8rem; font-weight: 800; color: #fff; }}
        .meta-info {{ font-size: 0.875rem; color: var(--text-muted); margin-top: 0.25rem; }}
        
        /* Hero Score Card */
        .hero-card {{
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .score-circle {{
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 8px solid {score_color};
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            background: rgba(15, 23, 42, 0.6);
        }}
        .score-num {{ font-size: 2.2rem; font-weight: 800; color: {score_color}; }}
        .score-label {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; }}
        
        .hero-details {{ flex: 1; }}
        .grade-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 700;
            background: {score_color}22;
            color: {score_color};
            border: 1px solid {score_color}66;
            margin-bottom: 0.75rem;
        }}
        .hero-stats {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        .stat-item {{ background: var(--surface); padding: 0.75rem 1.25rem; border-radius: 0.5rem; border: 1px solid var(--border); }}
        .stat-val {{ font-size: 1.2rem; font-weight: 700; color: #fff; }}
        .stat-lbl {{ font-size: 0.75rem; color: var(--text-muted); }}

        /* Alerts */
        .alerts-section {{ margin-bottom: 2.5rem; }}
        .section-title {{ font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; color: #fff; display: flex; align-items: center; gap: 0.5rem; }}
        .alert-item {{
            background: #EF444415;
            border-left: 4px solid var(--danger);
            padding: 0.85rem 1.25rem;
            border-radius: 0 0.5rem 0.5rem 0;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .alert-icon {{ color: var(--danger); font-size: 1.1rem; }}

        /* Dimensions Grid */
        .dim-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2.5rem;
        }}
        .dim-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
        }}
        .dim-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }}
        .dim-name {{ font-size: 0.9rem; font-weight: 600; color: var(--text); }}
        .dim-score {{ font-weight: 700; font-size: 0.95rem; }}
        .progress-bar-bg {{ background: #0F172A; height: 8px; border-radius: 4px; overflow: hidden; }}
        .progress-bar-fill {{ height: 100%; border-radius: 4px; }}
        .dim-footer {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; }}

        /* Table */
        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow-x: auto;
            margin-bottom: 2.5rem;
        }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.875rem; }}
        th {{ background: #0F172A; padding: 1rem; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }}
        td {{ padding: 0.85rem 1rem; border-bottom: 1px solid var(--border); }}
        tr:hover td {{ background: var(--surface-hover); }}
        .type-tag {{ background: #334155; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-family: monospace; }}
        .target-tag {{ background: #8B5CF6; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-left: 0.4rem; }}
        
        /* Badges */
        .badge {{ padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }}
        .badge-critical {{ background: #EF444433; color: #EF4444; border: 1px solid #EF444466; }}
        .badge-high {{ background: #F9731633; color: #F97316; border: 1px solid #F9731666; }}
        .badge-medium {{ background: #F59E0B33; color: #F59E0B; border: 1px solid #F59E0B66; }}
        .badge-low {{ background: #10B98133; color: #10B981; border: 1px solid #10B98166; }}

        /* Recommendations */
        .rec-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}
        .rec-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }}
        .rec-step {{ background: var(--primary); color: #fff; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }}
        .rec-title {{ font-size: 1rem; font-weight: 600; color: #fff; }}
        .rec-action {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem; }}
        pre {{
            background: #090D16;
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            border: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #38BDF8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <h1 class="title">Dataset Quality Audit Report</h1>
                <div class="meta-info">Audited Dataset: <strong>{escaped_ds_name}</strong> | ID: <code>{html.escape(report.audit_id)}</code> | Date: {html.escape(report.created_at)}</div>
            </div>
            <button onclick="window.print()" style="background: var(--primary); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 0.5rem; cursor: pointer; font-weight: 600;">Print / Save PDF</button>
        </div>

        <!-- Hero Score -->
        <div class="hero-card">
            <div class="score-circle">
                <div class="score-num">{report.overall_score}</div>
                <div class="score-label">Score / 100</div>
            </div>
            <div class="hero-details">
                <span class="grade-badge">Grade: {report.grade}</span>
                <h2 style="font-size: 1.4rem; color: #fff;">Dataset Quality Assessment</h2>
                <p style="color: var(--text-muted); font-size: 0.95rem;">
                    { 'Production ready! Dataset passed major quality checks.' if report.overall_score >= 90 else 'Action required. Multiple quality risks and ML hazards were detected.' }
                </p>
                <div class="hero-stats">
                    <div class="stat-item"><div class="stat-val">{report.total_rows:,}</div><div class="stat-lbl">Total Rows</div></div>
                    <div class="stat-item"><div class="stat-val">{report.total_columns}</div><div class="stat-lbl">Total Features</div></div>
                    <div class="stat-item"><div class="stat-val">{report.total_issues_count}</div><div class="stat-lbl">Identified Issues</div></div>
                    <div class="stat-item"><div class="stat-val">{report.memory_usage_mb} MB</div><div class="stat-lbl">Memory Footprint</div></div>
                </div>
            </div>
        </div>

        <!-- Summary Alerts -->
        <div class="alerts-section">
            <h3 class="section-title">Critical Summary Alerts</h3>
            {alerts_html}
        </div>

        <!-- Dimensions Breakdown -->
        <div>
            <h3 class="section-title">5-Dimension Quality Breakdown</h3>
            <div class="dim-grid">
                {dimensions_html}
            </div>
        </div>

        <!-- Column Profiles Table -->
        <div>
            <h3 class="section-title">Column Statistical Profiles ({len(report.column_profiles)} columns)</h3>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Inferred Type</th>
                            <th>Missing Values</th>
                            <th>Unique Values</th>
                            <th>Sample Data</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {col_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Actionable Recommendations -->
        <div>
            <h3 class="section-title">Recommended Remediation Action Plan ({len(report.recommended_actions)} steps)</h3>
            {recs_html}
        </div>
    </div>
</body>
</html>
"""
        return html_content
