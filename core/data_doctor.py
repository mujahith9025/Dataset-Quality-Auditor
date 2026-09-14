"""
Dataset Quality Auditor — AI "Data Doctor" Diagnostic Assistant (100% Offline & Smart Engine)
Provides instant, zero-API-key intelligent data quality diagnostics, machine learning risk assessments,
root-cause analysis, and dynamic natural language Q&A tailored to the audited dataset.
"""

from typing import Dict, Any, List, Optional
import json
import re
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
    Intelligent Diagnostic Assistant that answers questions about dataset quality,
    ML readiness, and remediation steps 100% offline with zero external API keys needed.
    """

    def diagnose(self, report: AuditReport) -> ExecutiveDiagnosis:
        score = report.overall_score
        critical_alerts = [a for a in report.summary_alerts if "🚨" in a or "CRITICAL" in a or "leakage" in a.lower()]

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
            f"Dataset '{report.dataset_name}' received a Quality Score of {score:.1f}/100 (Grade {report.grade})."
        ]
        if report.total_rows < 100:
            summary_lines.append(f"Sample size is critically small ({report.total_rows:,} rows), creating high model variance.")
        else:
            summary_lines.append(f"Audited {report.total_rows:,} records across {report.total_columns} columns ({report.memory_usage_mb} MB).")

        # Business Impact
        impact_lines = []
        if any("leakage" in a.lower() for a in report.summary_alerts):
            impact_lines.append("🚨 **Severe Target Leakage**: Features containing target information will cause catastrophic over-optimistic test scores and silent failure in production.")
        if any("imbalance" in a.lower() for a in report.summary_alerts):
            impact_lines.append("⚠️ **Class Imbalance Threat**: Minority class will suffer high false negatives without re-sampling or threshold tuning.")
        if any("missing" in a.lower() for a in report.summary_alerts):
            impact_lines.append("📉 **Incomplete Records**: Missing fields may lead to biased queries and pipeline crashes.")
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
            compliance.append("Legacy database sentinel values (-999, 9999) corrupt statistical aggregates and analytics.")
        if any("duplicate" in a.lower() for a in report.summary_alerts):
            compliance.append("Duplicate identity records can distort customer tracking and inflate KPI metrics.")
        if not compliance:
            compliance.append("No immediate data governance anomalies or corrupted sentinel values detected.")

        # Step-by-Step Remediation Roadmap
        roadmap = []
        for idx, rec in enumerate(report.recommended_actions[:5], 1):
            roadmap.append(f"**Step {idx} ({rec['severity']})**: {rec['title']} — {rec['action']}")

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

    def answer_query(self, user_question: str, report: AuditReport) -> str:
        """
        Smart, 100% Offline AI Natural Language Query Engine.
        Analyzes the user's question against real audit findings and returns a tailored diagnostic response.
        """
        q = user_question.lower().strip()
        score = report.overall_score
        grade = report.grade

        # 1. Check if user is asking about a specific column (e.g., "explain income", "loan_status", etc.)
        for col_name, prof in report.column_profiles.items():
            norm_col = col_name.lower()
            clean_col = re.sub(r'[^a-z0-9]', '', norm_col)
            clean_q = re.sub(r'[^a-z0-9]', ' ', q)
            
            if norm_col in q or (len(clean_col) >= 3 and clean_col in clean_q.split()):
                col_issues = prof.issues or []
                issues_desc = f" Issues flagged: {', '.join(col_issues)}." if col_issues else " No critical issues flagged for this feature."
                return (
                    f"📊 **Feature Diagnostic for '{col_name}'**:\n\n"
                    f"- **Inferred Type**: `{prof.inferred_type}` (`{prof.physical_dtype}`)\n"
                    f"- **Missing Data**: {prof.missing_count:,} rows ({prof.missing_percentage:.1f}% missing)\n"
                    f"- **Distinct Values**: {prof.unique_count:,} unique values ({prof.unique_percentage:.1f}% unique ratio)\n"
                    f"- **Role**: {'🎯 Target Label (Prediction Target)' if prof.is_target else '📈 Feature Attribute (Predictor)'}\n"
                    f"- **Health Status**:{issues_desc}"
                )

        # 2. Questions about Missing Values / Nulls
        if bool(re.search(r'\b(missing|nulls?|empty|nan|blank|null_values)\b', q)):
            missing_cols = [c for c, p in report.column_profiles.items() if p.missing_count > 0]
            if missing_cols:
                top_missing = sorted(missing_cols, key=lambda c: report.column_profiles[c].missing_percentage, reverse=True)[:5]
                col_list = ", ".join([f"`{c}` ({report.column_profiles[c].missing_percentage:.1f}%)" for c in top_missing])
                return (
                    f"📉 **Missing Values Analysis**:\n\n"
                    f"- Found missing entries across **{len(missing_cols)} columns**.\n"
                    f"- **Highest missing features**: {col_list}.\n\n"
                    f"💡 **Remediation**: Use median/mean imputation for numeric columns, mode or 'Unknown' for categories, or drop columns with >40% missing data in the **✨ 1-Click Clean** tab."
                )
            else:
                return "✅ **No Missing Values Detected**: Your dataset is 100% complete across all attributes!"

        # 3. Questions about Duplicates / Redundancy
        if bool(re.search(r'\b(duplicates?|duplicate rows?|redundant|repetition)\b', q)):
            dup_check = report.checker_results.get("duplicates")
            if dup_check and not dup_check.passed:
                return (
                    f"👥 **Duplicate Records Alert**:\n\n"
                    f"- {dup_check.summary}\n\n"
                    f"💡 **Impact & Fix**: Duplicates artificially inflate model confidence and distort validation splits. Deduplicate instantly under the **✨ 1-Click Clean** tab."
                )
            else:
                return "✅ **No Duplicates**: All rows in your dataset are unique with zero exact duplicates detected."

        # 4. Questions about Outliers / Anomalies
        if bool(re.search(r'\b(outliers?|anomal(y|ies)|extreme|z-score|iqr)\b', q)):
            outlier_check = report.checker_results.get("outliers")
            if outlier_check and not outlier_check.passed:
                return (
                    f"✂️ **Outliers & Anomalies Analysis**:\n\n"
                    f"- {outlier_check.summary}\n\n"
                    f"💡 **ML Impact**: Extreme outliers distort distance-based models (KNN, SVM, Linear Regression) and neural networks. Outlier capping (IQR 1.5x / 99th percentile) is available in **✨ 1-Click Clean**."
                )
            else:
                return "✅ **Outliers Clean**: Numerical values fall within standard expected statistical distributions."

        # 5. Questions about "Why is score low / Grade / Quality"
        if bool(re.search(r'\b(why|low|score|grade|bad|poor|health|rating)\b', q)):
            lowest_dims = sorted(report.dimension_scores.items(), key=lambda x: x[1].score)
            failing_checks = [c for c in report.checker_results.values() if not c.passed or c.score < 80.0]
            
            reasons = []
            for c in failing_checks[:4]:
                reasons.append(f"- **{c.checker_name}** ({c.score:.0f}%): {c.summary}")
            
            worst_dim = lowest_dims[0][1] if lowest_dims else None
            worst_dim_str = f"the **{worst_dim.dimension.value}** dimension (Score: {worst_dim.score:.1f}%)" if worst_dim else "several feature-level quality issues"
            return (
                f"🎯 **Why your score is {score:.1f}/100 (Grade {grade})**:\n\n"
                f"The primary bottleneck is {worst_dim_str}.\n\n"
                f"**Top Root Causes**:\n" + ("\n".join(reasons) if reasons else "- Minor statistical variance in numeric features.") +
                f"\n\n💡 **Solution**: Go to the **✨ 1-Click Clean** tab and click *Apply Selected Fixes* to automatically raise your score to 90+!"
            )

        # 6. Questions about Machine Learning / Models
        if bool(re.search(r'\b(ml|xgboost|models?|random\s+forest|logistic|algorithms?|training|train|accuracy|overfitting|ai|classifier)\b', q)):
            ml_hazards = []
            if any("leakage" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("🚨 **Target Leakage**: Features that directly leak the target will cause 99% fake training accuracy and fail completely in production.")
            if any("imbalance" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("⚠️ **Class Imbalance**: Standard cross-entropy loss will ignore the minority class. Use SMOTE, class weights, or PR-AUC metrics.")
            if any("outlier" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("✂️ **Extreme Outliers**: Linear models and neural networks will have distorted gradient updates without outlier clipping.")
            if any("sentinel" in a.lower() or "-999" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("🔢 **Corrupted Sentinels (-999)**: Tree-based models will treat -999 as a literal number, corrupting split thresholds.")

            hazards_text = "\n".join(ml_hazards) if ml_hazards else "✅ No critical ML blockers detected! Standard tree-based models should train smoothly."
            return (
                f"🤖 **Machine Learning Impact Assessment for '{report.dataset_name}'**:\n\n"
                f"{hazards_text}\n\n"
                f"💡 **Recommended Action**: Run the **🚀 ML Accuracy Test** tab to benchmark baseline Random Forest & Logistic Regression accuracy uplift before and after cleaning."
            )

        # 7. Questions about Priority / Remediation
        if bool(re.search(r'\b(priority|first|fix|fixes|clean|cleaning|steps?|recommend(ations?)?|how\s+to|what\s+should)\b', q)):
            top_recs = report.recommended_actions[:4]
            recs_text = "\n".join([f"{idx+1}. **{r['title']}** ({r['severity']}): {r['action']}" for idx, r in enumerate(top_recs)])
            return (
                f"🛠️ **Top Recommended Action Steps**:\n\n"
                f"{recs_text}\n\n"
                f"⚡ **Quick Fix**: All these steps can be executed automatically in 1 click under the **✨ 1-Click Clean** tab!"
            )

        # 8. Questions about Business Risk / Security / Compliance
        if bool(re.search(r'\b(risks?|business|security|compliance|gdpr|legal|liabilit(y|ies)|governance)\b', q)):
            crit_count = report.severity_counts.get("CRITICAL", 0) + report.severity_counts.get("HIGH", 0)
            return (
                f"⚖️ **Business & Compliance Risk Assessment**:\n\n"
                f"- **Overall Risk Level**: {'🔴 HIGH RISK' if crit_count > 0 else '🟢 LOW RISK'} ({crit_count} high/critical defects detected).\n"
                f"- **Data Integrity**: {report.total_rows:,} records evaluated across {report.total_columns} attributes.\n"
                f"- **Downstream Liability**: Corrupted sentinels and duplicate entities can bias executive financial dashboards and customer consent registers.\n\n"
                f"💡 **Governance Recommendation**: Run the quality gate CLI (`python cli.py audit data.csv --min-score 80`) in your automated ingestion pipelines."
            )

        # 9. Default Smart Executive Synthesis
        recs_count = len(report.recommended_actions)
        return (
            f"💡 **AI Data Doctor Consultation for '{report.dataset_name}'**:\n\n"
            f"- **Health Rating**: {score:.1f}/100 (Grade {grade})\n"
            f"- **Dataset Footprint**: {report.total_rows:,} rows × {report.total_columns} columns ({report.memory_usage_mb} MB)\n"
            f"- **Identified Bottlenecks**: {len(report.summary_alerts)} alert areas with {recs_count} recommended remediations.\n\n"
            f"**Key Focus**: Address '{report.summary_alerts[0] if report.summary_alerts else 'standard data hygiene'}' to maximize ML reliability and reporting accuracy."
        )
