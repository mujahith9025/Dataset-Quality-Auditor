"""
Dataset Quality Auditor — AI "Data Doctor" Executive Summary & Assistant
Generates executive-level diagnostic briefings, business risk assessments, root-cause analyses,
and provides optional Gemini / LLM natural language Q&A for non-technical stakeholders.
"""

from typing import Dict, Any, List, Optional
import urllib.request
import json
from pydantic import BaseModel, Field
from models.schemas import AuditReport


class ExecutiveDiagnosis(BaseModel):
    overall_health_badge: str
    risk_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    risk_score: int  # 0 to 100
    executive_summary: str
    business_impact: str
    root_cause_vectors: List[str]
    compliance_risks: List[str]
    remediation_roadmap: List[str]


class DataDoctor:
    """
    Synthesizes multi-dimension audit findings into plain-English business & engineering diagnostics.
    """

    def diagnose(self, report: AuditReport) -> ExecutiveDiagnosis:
        score = report.overall_score
        critical_alerts = [a for a in report.summary_alerts if "🚨" in a or "CRITICAL" in a]
        warning_alerts = [a for a in report.summary_alerts if "⚠️" in a]

        # Determine Risk Level
        if score >= 85 and len(critical_alerts) == 0:
            risk_level = "LOW"
            risk_score = 15
            badge = "🟢 PRODUCTION READY"
        elif score >= 70:
            risk_level = "MODERATE"
            risk_score = 45
            badge = "🟡 OBSERVATION REQUIRED"
        elif score >= 50 or len(critical_alerts) > 0:
            risk_level = "HIGH"
            risk_score = 75
            badge = "🟠 HIGH RISK FOR ML DEPLOYMENT"
        else:
            risk_level = "CRITICAL"
            risk_score = 95
            badge = "🔴 BLOCKED / COMPROMISED DATASET"

        # Executive Summary Synthesis
        summary_lines = [
            f"Dataset '{report.dataset_name}' received an overall Quality Score of {score:.1f}/100 (Grade {report.grade})."
        ]
        if report.total_rows < 100:
            summary_lines.append(f"Sample size is critically small ({report.total_rows:,} rows), exposing any downstream models to extreme variance.")
        else:
            summary_lines.append(f"Audit inspected {report.total_rows:,} records across {report.total_columns} attributes ({report.memory_usage_mb} MB).")

        # Business Impact
        impact_lines = []
        if any("leakage" in a.lower() for a in report.summary_alerts):
            impact_lines.append("🚨 **Severe Target Leakage Risk**: Feature(s) containing target information will cause catastrophic failure in production inference.")
        if any("imbalance" in a.lower() for a in report.summary_alerts):
            impact_lines.append("⚠️ **Class Imbalance Threat**: Minority class will suffer high false negatives without re-sampling or cost-sensitive training.")
        if any("missing" in a.lower() for a in report.summary_alerts):
            impact_lines.append("📉 **Incomplete Records**: Missing fields may lead to biased business intelligence queries and data pipeline crashes.")
        if not impact_lines:
            impact_lines.append("✅ **Low Business Liability**: The dataset shows consistent records and is well-suited for automated ML pipelines.")

        # Root Cause Vectors
        root_causes = []
        for check in report.checker_results.values():
            if not check.passed or check.score < 95.0:
                root_causes.append(f"**{check.checker_name}**: {check.summary}")

        # Compliance & Governance Risks
        compliance = []
        if any("sentinel" in a.lower() or "-999" in a.lower() for a in report.summary_alerts):
            compliance.append("Legacy database sentinel values (-999, 9999) corrupt statistical aggregates and regulatory audit logs.")
        if any("duplicate" in a.lower() for a in report.summary_alerts):
            compliance.append("Duplicate identity records can distort customer consent tracking and inflate KPI metrics.")
        if not compliance:
            compliance.append("No immediate data governance anomalies or corrupted sentinel values detected.")

        # Step-by-Step Remediation Roadmap
        roadmap = []
        for idx, rec in enumerate(report.recommended_actions[:5], 1):
            roadmap.append(f"**Phase {idx} ({rec['severity']})**: {rec['title']} — {rec['action']}")

        return ExecutiveDiagnosis(
            overall_health_badge=badge,
            risk_level=risk_level,
            risk_score=risk_score,
            executive_summary=" ".join(summary_lines),
            business_impact=" ".join(impact_lines),
            root_cause_vectors=root_causes[:6],
            compliance_risks=compliance,
            remediation_roadmap=roadmap
        )

    def ask_gemini(self, user_question: str, report: AuditReport, api_key: str) -> str:
        """
        Queries Google Gemini Generative AI model with the dataset quality context.
        """
        if not api_key or not api_key.strip():
            return "Please provide a valid Google Gemini API Key to enable interactive generative AI diagnostics."

        system_context = (
            f"You are the Chief Data Officer & AI Data Doctor. You are analyzing an audit report for dataset '{report.dataset_name}'. "
            f"Overall Quality Score: {report.overall_score}/100 (Grade: {report.grade}). "
            f"Dimensions: {json.dumps({k: v.score for k, v in report.dimension_scores.items()})}. "
            f"Key Alerts: {'; '.join(report.summary_alerts)}. "
            f"Answer the user's inquiry concisely with practical engineering and business recommendations."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_context}\n\nUser Question: {user_question}"}
                    ]
                }
            ]
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            return f"Gemini API Request Error: {str(e)}"
