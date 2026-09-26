"""
Form16 Analyzer — upload and automatically extract Form 16 data.
Automatically propagates extracted data to session state so Tax Calculator,
Tax Optimizer, and Regime Comparator can pre-fill from Form 16.
"""
import os
import tempfile
import streamlit as st
from frontend.components.cards import metric_card


def _int(val, default=0) -> int:
    """Safely cast to int, fallback to default."""
    try:
        return int(val) if val else default
    except (TypeError, ValueError):
        return default


def _store_form16_in_session(form16_data) -> dict:
    """
    Convert Form16Data → tax profile dict and store in session state.
    Uses the SAME key names as tax_calculator so optimizer/comparator pick them up.
    Returns the stored profile dict.
    """
    b = form16_data.part_b

    # 80C breakdown: parser only gives us an aggregate 80C amount.
    # We store it in other_80c so it shows up correctly everywhere.
    total_80c = _int(b.sec_80c)

    profile = {
        # Income
        "gross_salary":            _int(b.gross_salary),
        "basic_salary":            _int(b.basic_salary) or _int(b.gross_salary * 0.5),
        "hra_received":            _int(b.hra_received),
        "rent_paid":               0,           # not in Form 16 directly
        "is_metro":                False,       # unknown from Form 16
        "other_income":            0,

        # 80C components — store total in other_80c
        "elss":                    0,
        "ppf":                     0,
        "epf":                     0,
        "life_insurance":          0,
        "home_loan_principal":     0,
        "nsc":                     0,
        "other_80c":               total_80c,

        # 80D
        "health_insurance_self":   _int(b.sec_80d),
        "health_insurance_parents": 0,
        "self_age_above_60":       False,
        "parent_age_above_60":     False,

        # NPS
        "additional_nps_80ccd1b":  _int(b.sec_80ccd1b),
        "employer_nps":            _int(b.employer_nps_80ccd2),
        "is_govt_employer":        False,

        # Other deductions
        "savings_interest":        _int(b.sec_80tta),
        "home_loan_interest":      0,
        "is_self_occupied":        True,
    }

    st.session_state["last_tax_profile"]   = profile
    st.session_state["form16_loaded"]      = True
    st.session_state["form16_employer"]    = form16_data.part_a.employer_name or ""
    st.session_state["form16_ay"]          = form16_data.part_a.assessment_year or "2026-27"
    # Clear old tax result so it gets recalculated with the new profile
    st.session_state.pop("last_tax_result", None)
    return profile


def show_form16_analyzer():
    st.markdown('<p class="section-header"> Form16 Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload your Form 16 PDF — data auto-fills Tax Calculator and Tax Optimizer</p>', unsafe_allow_html=True)

    # Banner if Form 16 already loaded this session
    if st.session_state.get("form16_loaded"):
        emp = st.session_state.get("form16_employer", "")
        ay  = st.session_state.get("form16_ay", "")
        st.success(
            f"Form 16 loaded from **{emp}** (AY {ay}) — "
            "Tax Calculator and Tax Optimizer are pre-filled. Upload a new file to replace."
        )

    st.info(
        "**How it works:** Upload your Form 16 PDF below. The system will automatically extract "
        "salary, deductions, and TDS details — then pre-fill the Tax Calculator and Tax Optimizer "
        "so you can run a full analysis without re-entering data."
    )

    uploaded = st.file_uploader(
        "Drop your Form 16 PDF here",
        type=["pdf"],
        key="form16_main_upload",
        help="Form 16 is issued by your employer. It contains your salary, TDS, and deductions for the financial year.",
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner(" Extracting Form 16 data..."):
            try:
                from form16_parser.parser import extract_form16
                form16_data = extract_form16(tmp_path)
                b = form16_data.part_b
                a = form16_data.part_a
            except Exception as e:
                st.error(f"Error parsing Form 16: {e}")
                os.unlink(tmp_path)
                return

        os.unlink(tmp_path)

        # Auto-store profile in session state immediately on upload
        profile = _store_form16_in_session(form16_data)

        # Confidence indicator
        confidence = form16_data.extraction_confidence or 0
        conf_color = "#4ECDC4" if confidence > 0.6 else "#FFD700" if confidence > 0.3 else "#FF6B6B"
        conf_label = (
            "Good extraction — data pre-filled in Tax Calculator & Optimizer"
            if confidence > 0.6
            else "Partial extraction — verify values before analysis"
            if confidence > 0.3
            else "Low confidence — please verify all values manually"
        )
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3); border-radius:8px; padding:12px;
                    border:1px solid {conf_color}; margin-bottom:16px;">
            <span style="color:{conf_color}; font-weight:600;">
                Extraction Confidence: {confidence*100:.0f}%
            </span>
            <span style="color:#8B949E; font-size:13px; margin-left:12px;">{conf_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Employer info
        if a.employer_name:
            st.markdown(
                f"**Employer:** {a.employer_name} &nbsp;|&nbsp; "
                f"**TAN:** {a.employer_tan or 'N/A'} &nbsp;|&nbsp; "
                f"**AY:** {a.assessment_year}"
            )

        # KPI cards
        st.markdown("#### Income Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Gross Salary", f"Rs.{b.gross_salary:,.0f}", "", "#4ECDC4", "")
        with col2:
            metric_card("TDS Deducted", f"Rs.{b.tds_deducted:,.0f}", "", "#FF6B6B", "")
        with col3:
            metric_card("Taxable Income", f"Rs.{b.taxable_income:,.0f}", "", "#FFD700", "")
        with col4:
            metric_card("Total Tax Payable", f"Rs.{b.total_tax_payable:,.0f}", "", "#A78BFA", "")

        st.divider()

        # Deductions + Tax side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Chapter VI-A Deductions")
            rows = [
                ("Section 80C",            b.sec_80c),
                ("Section 80D",            b.sec_80d),
                ("80CCD(1B) — Add. NPS",   b.sec_80ccd1b),
                ("80CCD(2) — Employer NPS", b.employer_nps_80ccd2),
                ("Section 80E",            b.sec_80e),
                ("Section 80G",            b.sec_80g),
                ("Section 80TTA",          b.sec_80tta),
            ]
            for label, val in rows:
                color = "#4ECDC4" if val and val > 0 else "#64748b"
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<span style="color:#94a3b8;">{label}</span>'
                    f'<span style="color:{color}; font-weight:600;">Rs.{(val or 0):,.0f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; '
                f'padding:6px 0; margin-top:4px;">'
                f'<span style="color:#f1f5f9; font-weight:700;">Total Deductions</span>'
                f'<span style="color:#FFD700; font-weight:700;">Rs.{(b.total_deductions_chapter_via or 0):,.0f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown("#### Tax Computation")
            tax_rows = [
                ("Tax on Income",      b.tax_on_income),
                ("Rebate u/s 87A",     b.rebate_87a),
                ("Surcharge",          b.surcharge),
                ("H&E Cess (4%)",      b.cess),
            ]
            for label, val in tax_rows:
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<span style="color:#94a3b8;">{label}</span>'
                    f'<span style="color:#e2e8f0; font-weight:600;">Rs.{(val or 0):,.0f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; '
                f'padding:6px 0; margin-top:4px;">'
                f'<span style="color:#f1f5f9; font-weight:700;">Total Tax Payable</span>'
                f'<span style="color:#FF6B6B; font-weight:700;">Rs.{(b.total_tax_payable or 0):,.0f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            refund = (b.tds_deducted or 0) - (b.total_tax_payable or 0)
            if refund > 0:
                st.success(f" Refund Due: Rs.{refund:,.0f}")
            elif refund < 0:
                st.warning(f"Additional Tax Due: Rs.{abs(refund):,.0f}")
            else:
                st.info("TDS matches tax payable — no refund or additional tax")

        # Parsing notes
        if form16_data.parsing_notes:
            st.warning("**Extraction Notes:**\n" + "\n".join(f"- {n}" for n in form16_data.parsing_notes))

        st.divider()

        # Pre-filled profile summary
        st.markdown("#### Pre-filled Profile (ready for Tax Calculator & Optimizer)")
        st.caption(
            "The following values have been stored in session state. "
            "Switch to **Tax Calculator** and click **Calculate Tax**, "
            "or go to **Tax Optimizer** to see optimization opportunities."
        )
        with st.expander("View pre-filled profile", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"- **Gross Salary:** Rs.{profile['gross_salary']:,}")
                st.markdown(f"- **Basic Salary:** Rs.{profile['basic_salary']:,}")
                st.markdown(f"- **HRA Received:** Rs.{profile['hra_received']:,}")
                st.markdown(f"- **80C (aggregate):** Rs.{profile['other_80c']:,}")
                st.markdown(f"- **Health Ins. (Self):** Rs.{profile['health_insurance_self']:,}")
            with col2:
                st.markdown(f"- **80CCD(1B) NPS:** Rs.{profile['additional_nps_80ccd1b']:,}")
                st.markdown(f"- **Employer NPS 80CCD(2):** Rs.{profile['employer_nps']:,}")
                st.markdown(f"- **80TTA Savings Interest:** Rs.{profile['savings_interest']:,}")

        # One-click run full analysis
        if b.gross_salary and b.gross_salary > 0:
            if st.button(" Run Full Tax Analysis Now", key="form16_analyze_btn", use_container_width=True):
                try:
                    from tax_engine.calculator import calculate_full_tax
                    result = calculate_full_tax(profile)
                    st.session_state["last_tax_result"] = result
                    old_tax = result["old_regime"]["total_tax"]
                    new_tax = result["new_regime"]["total_tax"]
                    rec     = result["recommendation"]["regime"]
                    st.success(
                        f"Analysis complete! Old Regime: Rs.{old_tax:,.0f} | "
                        f"New Regime: Rs.{new_tax:,.0f} | Recommended: **{rec} Regime**  "
                        f"Now switch to **Tax Optimizer** for savings opportunities."
                    )
                except Exception as e:
                    st.error(f"Analysis error: {e}")
