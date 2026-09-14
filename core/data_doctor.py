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
            badge = "🟠 HIGH RISK FOR ML"
        else:
            risk_level = "CRITICAL"
            risk_score = 95
            badge = "🔴 COMPROMISED DATASET"

        # Crisp Executive Summary
        summary = f"Quality Score: {score:.1f}/100 (Grade {report.grade}) across {report.total_rows:,} rows and {report.total_columns} columns."

        # Crisp Business Impact
        impact_lines = []
        if any("leakage" in a.lower() for a in report.summary_alerts):
            impact_lines.append("🚨 **Target Leakage**: Gives away predictions, causing false high accuracy.")
        if any("imbalance" in a.lower() for a in report.summary_alerts):
            impact_lines.append("⚠️ **Class Imbalance**: Minority class will be ignored without re-sampling.")
        if any("missing" in a.lower() for a in report.summary_alerts):
            impact_lines.append("📉 **Missing Data**: Incomplete records will cause errors or bias.")
        if not impact_lines:
            impact_lines.append("✅ **Clean & Reliable**: Ready for analytics and ML models.")

        # Root Cause Vectors (Crisp 1-liners)
        root_causes = []
        for check in report.checker_results.values():
            if not check.passed or check.score < 95.0:
                root_causes.append(f"**{check.checker_name}**: {check.summary}")

        # Governance Risks
        compliance = []
        if any("sentinel" in a.lower() or "-999" in a.lower() for a in report.summary_alerts):
            compliance.append("Legacy error tokens (-999, 9999) corrupt calculations.")
        if any("duplicate" in a.lower() for a in report.summary_alerts):
            compliance.append("Duplicate rows distort customer counts and KPIs.")
        if not compliance:
            compliance.append("No governance or sentinel errors detected.")

        # Crisp Remediation Roadmap
        roadmap = []
        for idx, rec in enumerate(report.recommended_actions[:4], 1):
            roadmap.append(f"**Step {idx}**: {rec['title']} — {rec['action']}")

        return ExecutiveDiagnosis(
            overall_health_badge=badge,
            risk_level=risk_level,
            risk_score=risk_score,
            executive_summary=summary,
            business_impact=" ".join(impact_lines),
            root_cause_vectors=root_causes[:5],
            compliance_risks=compliance,
            remediation_roadmap=roadmap
        )

    def answer_query(self, user_question: str, report: AuditReport) -> str:
        """
        Smart, 100% Offline AI Natural Language Query Engine.
        Returns crisp, simple, beginner-friendly diagnostic answers.
        """
        q = user_question.lower().strip()
        score = report.overall_score
        grade = report.grade

        # 1. Feature-specific inquiry
        for col_name, prof in report.column_profiles.items():
            norm_col = col_name.lower()
            clean_col = re.sub(r'[^a-z0-9]', '', norm_col)
            clean_q = re.sub(r'[^a-z0-9]', ' ', q)
            
            if norm_col in q or (len(clean_col) >= 3 and clean_col in clean_q.split()):
                col_issues = prof.issues or []
                status_text = f"⚠️ {', '.join(col_issues)}" if col_issues else "✅ Clean (No issues)"
                return (
                    f"📊 **Feature '{col_name}' Profile**:\n\n"
                    f"- **Type**: `{prof.inferred_type}`\n"
                    f"- **Missing**: {prof.missing_count:,} rows ({prof.missing_percentage:.1f}%)\n"
                    f"- **Unique**: {prof.unique_count:,} distinct values\n"
                    f"- **Role**: {'🎯 Target Label' if prof.is_target else '📈 Feature'}\n"
                    f"- **Status**: {status_text}"
                )

        # 2. Missing Values
        if bool(re.search(r'\b(missing|nulls?|empty|nan|blank|null_values)\b', q)):
            missing_cols = [c for c, p in report.column_profiles.items() if p.missing_count > 0]
            if missing_cols:
                top_missing = sorted(missing_cols, key=lambda c: report.column_profiles[c].missing_percentage, reverse=True)[:4]
                col_list = ", ".join([f"`{c}` ({report.column_profiles[c].missing_percentage:.1f}%)" for c in top_missing])
                return (
                    f"📉 **Missing Values**:\n\n"
                    f"- Found in **{len(missing_cols)} columns**: {col_list}\n"
                    f"- **Quick Fix**: Auto-impute with median or mode under **✨ 1-Click Clean**."
                )
            return "✅ **No Missing Values**: All records are 100% complete!"

        # 3. Duplicates
        if bool(re.search(r'\b(duplicates?|duplicate rows?|redundant|repetition)\b', q)):
            dup_check = report.checker_results.get("duplicates")
            if dup_check and not dup_check.passed:
                return f"👥 **Duplicate Rows**:\n\n- {dup_check.summary}\n- **Fix**: Remove duplicate rows in 1-click in **✨ 1-Click Clean**."
            return "✅ **No Duplicates**: All rows are unique."

        # 4. Outliers
        if bool(re.search(r'\b(outliers?|anomal(y|ies)|extreme|z-score|iqr)\b', q)):
            outlier_check = report.checker_results.get("outliers")
            if outlier_check and not outlier_check.passed:
                return f"✂️ **Outliers**:\n\n- {outlier_check.summary}\n- **Fix**: Clip extreme values (IQR 1.5x) in **✨ 1-Click Clean**."
            return "✅ **Outliers Clean**: Numerical values follow normal ranges."

        # 5. Why is score low / Grade
        if bool(re.search(r'\b(why|low|score|grade|bad|poor|health|rating)\b', q)):
            lowest_dims = sorted(report.dimension_scores.items(), key=lambda x: x[1].score)
            failing = [c for c in report.checker_results.values() if not c.passed or c.score < 80.0]
            reasons = [f"- **{c.checker_name}** ({c.score:.0f}%): {c.summary}" for c in failing[:3]]
            worst_name = lowest_dims[0][1].dimension.value if lowest_dims else "general quality"
            
            return (
                f"🎯 **Score Breakdown ({score:.1f}/100 • Grade {grade})**:\n\n"
                f"- **Main Bottleneck**: {worst_name}\n"
                f"- **Top Causes**:\n" + ("\n".join(reasons) if reasons else "- Minor numeric variations") + "\n\n"
                f"💡 **Fix**: Click *Apply Selected Fixes* in **✨ 1-Click Clean** to boost your score to 90+!"
            )

        # 6. Machine Learning Impact
        if bool(re.search(r'\b(ml|xgboost|models?|random\s+forest|logistic|algorithms?|training|train|accuracy|overfitting|ai|classifier)\b', q)):
            ml_hazards = []
            if any("leakage" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("- 🚨 **Target Leakage**: Causes fake 99% accuracy and production failure.")
            if any("imbalance" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("- ⚠️ **Class Imbalance**: Standard models will ignore the minority class.")
            if any("outlier" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("- ✂️ **Outliers**: Distorts linear models and neural networks.")
            if any("sentinel" in a.lower() or "-999" in a.lower() for a in report.summary_alerts):
                ml_hazards.append("- 🔢 **Corrupted Numbers (-999)**: Causes bad decision splits.")

            hazards = "\n".join(ml_hazards) if ml_hazards else "✅ No ML blockers detected. Data is ready to train!"
            return (
                f"🤖 **Machine Learning Impact**:\n\n"
                f"{hazards}\n\n"
                f"💡 **Test Uplift**: Go to **🚀 ML Accuracy Test** to see accuracy before and after cleaning."
            )

        # 7. Priority & Remediation
        if bool(re.search(r'\b(priority|first|fix|fixes|clean|cleaning|steps?|recommend(ations?)?|how\s+to|what\s+should)\b', q)):
            top_recs = report.recommended_actions[:3]
            recs_text = "\n".join([f"{idx+1}. **{r['title']}**: {r['action']}" for idx, r in enumerate(top_recs)])
            return (
                f"🛠️ **Top Priority Fixes**:\n\n"
                f"{recs_text}\n\n"
                f"⚡ **Quick Action**: Apply all in 1-click in the **✨ 1-Click Clean** tab."
            )

        # 8. Business & Compliance Risk
        if bool(re.search(r'\b(risks?|business|security|compliance|gdpr|legal|liabilit(y|ies)|governance)\b', q)):
            crit_count = report.severity_counts.get("CRITICAL", 0) + report.severity_counts.get("HIGH", 0)
            return (
                f"⚖️ **Business Risk Summary**:\n\n"
                f"- **Risk Rating**: {'🔴 HIGH RISK' if crit_count > 0 else '🟢 LOW RISK'} ({crit_count} high-severity defects)\n"
                f"- **Data Footprint**: {report.total_rows:,} rows × {report.total_columns} columns\n"
                f"- **Recommendation**: Clean data before generating executive metrics or training AI."
            )

        # 9. Default Summary
        return (
            f"💡 **Dataset Quality Snapshot**:\n\n"
            f"- **Health Score**: {score:.1f}/100 (Grade {grade})\n"
            f"- **Dimensions Checked**: 5 Quality Pillars across 9 automated tests\n"
            f"- **Key Action**: {report.summary_alerts[0] if report.summary_alerts else 'Apply 1-Click Clean to optimize dataset'}"
        )
