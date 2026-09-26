"""
Tax Optimizer page - identifies unused deductions and quantifies savings.
"""
import streamlit as st
from frontend.components.charts import (
    optimization_horizontal_bar, tax_savings_gauge,
    opp_stacked_bar, opp_treemap, savings_breakdown_donut,
)
from frontend.components.cards import metric_card, priority_badge


def show_tax_optimizer():
    st.markdown('<p class="section-header"> Tax Optimizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Discover missed deductions and maximize your tax savings</p>', unsafe_allow_html=True)

    has_profile = "last_tax_profile" in st.session_state and "last_tax_result" in st.session_state
    form16_loaded = st.session_state.get("form16_loaded", False)

    if has_profile:
        if form16_loaded:
            emp = st.session_state.get("form16_employer", "")
            st.info(f"Profile pre-filled from Form 16 ({emp}). Values below are editable — click Analyze to run.")
        else:
            st.info("Using profile from Tax Calculator. You can also enter values below to analyze a different scenario.")

    def _ss_int(key: str, subkey: str, default: int) -> int:
        """Get int from nested session_state profile, always returns int."""
        return int(st.session_state.get(key, {}).get(subkey, default) or default)

    with st.expander("Enter / Edit Profile for Optimization", expanded=not has_profile):
        col1, col2, col3 = st.columns(3)
        with col1:
            gross_salary = st.number_input(
                "Gross Salary (Rs.)", min_value=0,
                value=_ss_int("last_tax_profile", "gross_salary", 1_200_000),
                step=50_000, format="%d", key="opt_salary")
        with col2:
            used_80c = st.number_input(
                "Total 80C Invested (Rs.)", min_value=0,
                value=int(sum(
                    _ss_int("last_tax_profile", k, 0)
                    for k in ["elss", "ppf", "epf", "life_insurance", "home_loan_principal", "nsc", "other_80c"]
                )),
                max_value=150_000, step=5_000, format="%d", key="opt_80c")
        with col3:
            health_self = st.number_input(
                "Health Ins. (Self) (Rs.)", min_value=0,
                value=_ss_int("last_tax_profile", "health_insurance_self", 0),
                max_value=50_000, step=1_000, format="%d", key="opt_80d_self")

        col1, col2, col3 = st.columns(3)
        with col1:
            health_parents = st.number_input(
                "Health Ins. (Parents) (Rs.)", min_value=0,
                value=_ss_int("last_tax_profile", "health_insurance_parents", 0),
                max_value=50_000, step=1_000, format="%d", key="opt_80d_par")
        with col2:
            additional_nps = st.number_input(
                "Additional NPS 80CCD(1B) (Rs.)", min_value=0,
                value=_ss_int("last_tax_profile", "additional_nps_80ccd1b", 0),
                max_value=50_000, step=5_000, format="%d", key="opt_nps")
        with col3:
            parent_senior = st.checkbox("Parents age 60+?", value=False, key="opt_parent_senior")

        analyze_btn = st.button("Analyze Optimization Opportunities", key="opt_analyze", use_container_width=True)

    if analyze_btn or has_profile:
        # Build profile: prefer session_state (from Form16/Calculator), merge with widget values
        base_profile = st.session_state.get("last_tax_profile", {})
        profile = {
            **base_profile,
            # Override with current widget values (user may have edited them)
            "gross_salary":             int(gross_salary),
            "other_80c":                int(used_80c),
            "health_insurance_self":    int(health_self),
            "health_insurance_parents": int(health_parents),
            "additional_nps_80ccd1b":   int(additional_nps),
            "parent_age_above_60":      parent_senior,
        }

        with st.spinner("Finding optimization opportunities..."):
            try:
                from tax_engine.calculator import calculate_full_tax
                from optimization.optimizer import find_optimization_opportunities

                if "last_tax_result" not in st.session_state:
                    st.session_state["last_tax_result"] = calculate_full_tax(profile)

                opt_result = find_optimization_opportunities(
                    profile=profile,
                    tax_result=st.session_state["last_tax_result"],
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return

        # Summary KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            old_tax = st.session_state["last_tax_result"]["old_regime"]["total_tax"]
            metric_card("Current Old Regime Tax", f"Rs.{old_tax:,.0f}", "", "#FF6B6B", "")
        with col2:
            new_tax = st.session_state["last_tax_result"]["new_regime"]["total_tax"]
            metric_card("Current New Regime Tax", f"Rs.{new_tax:,.0f}", "", "#4ECDC4", "")
        with col3:
            pot_savings = opt_result["total_potential_additional_savings"]
            metric_card("Additional Savings Possible", f"Rs.{pot_savings:,.0f}", "Under Old Regime", "#FFD700", "")

        st.divider()
        st.markdown(f"### {opt_result['summary']}")

        # Savings breakdown donut (full width, no gauge)
        if opt_result["opportunities"] and pot_savings > 0:
            st.plotly_chart(
                savings_breakdown_donut(opt_result["opportunities"]),
                use_container_width=True,
            )

        st.divider()

        # Interactive Charts: Stacked Bar + Treemap in tabs
        # if opt_result["opportunities"]:
        #     st.markdown("### Optimization Charts")
        #     tab1, tab2 = st.tabs(["Deduction Utilisation", "Savings Map"])

        #     with tab1:
        #         st.caption(
        #             "Each bar shows how much of a deduction limit you have used (solid = priority colour) "
        #             "vs how much remains as a gap. Hover for exact amounts."
        #         )
        #         st.plotly_chart(
        #             opp_stacked_bar(opt_result["opportunities"]),
        #             use_container_width=True,
        #         )

        #     with tab2:
        #         st.caption(
        #             "Each block is sized by estimated tax savings. "
        #             "RED = HIGH priority, AMBER = MEDIUM, TEAL = LOW. Hover for section details."
        #         )
        #         st.plotly_chart(
        #             opp_treemap(opt_result["opportunities"]),
        #             use_container_width=True,
        #         )

        st.divider()
        st.markdown("### Optimization Opportunities")

        priority_bg = {"HIGH": "#FF6B6B22", "MEDIUM": "#FFD70022", "LOW": "#4ECDC422"}
        priority_border = {"HIGH": "#FF6B6B", "MEDIUM": "#FFD700", "LOW": "#4ECDC4"}
        priority_label_color = {"HIGH": "#FF6B6B", "MEDIUM": "#FFD700", "LOW": "#4ECDC4"}

        for opp in opt_result["opportunities"]:
            pri = opp.get("priority", "MEDIUM")
            current = opp.get("current_investment", 0)
            limit   = opp.get("limit", 0)
            gap     = opp.get("remaining_capacity", 0)
            savings = opp.get("estimated_tax_savings", 0)
            pct_used = int(current / limit * 100) if limit > 0 else 0
            instruments = ", ".join(opp.get("instruments", []))
            bar_empty  = "rgba(255,255,255,0.08)"

            card_html = """
            <div style="
                background:{bg};
                border:1px solid {border};
                border-left:4px solid {border};
                border-radius:10px;
                padding:18px 22px 14px 22px;
                margin-bottom:14px;
            ">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                    <div style="flex:1; min-width:200px;">
                        <span style="font-size:1.02rem; font-weight:600; color:#f1f5f9;">
                            {title}
                        </span>
                        <span style="font-size:0.8rem; color:#94a3b8; margin-left:8px;">
                            Section {section}
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:14px; flex-shrink:0;">
                        <span style="font-size:1.05rem; font-weight:700; color:#FFD700;">
                            Rs.{savings} saved
                        </span>
                        <span style="
                            background:{label_bg};
                            color:{label_color};
                            border:1px solid {border};
                            border-radius:20px; padding:2px 12px;
                            font-size:0.75rem; font-weight:700; letter-spacing:0.05em;
                        ">{pri}</span>
                    </div>
                </div>
                <div style="margin:12px 0 6px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:0.78rem; color:#94a3b8;">
                            Invested: Rs.{current} / Rs.{limit}
                        </span>
                        <span style="font-size:0.78rem; color:#94a3b8;">{pct}% used</span>
                    </div>
                    <div style="background:{bar_empty}; border-radius:6px; height:8px; overflow:hidden;">
                        <div style="
                            width:{pct}%; height:100%;
                            background:{bar_color};
                            border-radius:6px;
                        "></div>
                    </div>
                    <div style="font-size:0.78rem; color:#FF6B6B; margin-top:4px;">
                        Gap remaining: Rs.{gap}
                    </div>
                </div>
                <div style="font-size:0.8rem; color:#cbd5e1; margin-top:6px;">
                    <span style="color:#64748b; font-weight:600;">Instruments: </span>{instruments}
                </div>
            </div>
            """.format(
                bg=priority_bg.get(pri, "#ffffff11"),
                border=priority_border.get(pri, "#ffffff33"),
                label_bg=priority_border.get(pri, "#4ECDC4") + "33",
                label_color=priority_label_color.get(pri, "#4ECDC4"),
                title=opp["title"],
                section=opp.get("section", ""),
                savings=f"{savings:,.0f}",
                pri=pri,
                current=f"{current:,.0f}",
                limit=f"{limit:,.0f}",
                pct=pct_used,
                bar_empty=bar_empty,
                bar_color=priority_border.get(pri, "#4ECDC4"),
                gap=f"{gap:,.0f}",
                instruments=instruments,
            )
            st.markdown(card_html, unsafe_allow_html=True)