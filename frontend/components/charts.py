"""
Plotly chart components for FinAssist AI.
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List


#  Color Palette 
OLD_REGIME_COLOR = "#FF6B6B"
NEW_REGIME_COLOR = "#4ECDC4"
SAVINGS_COLOR = "#45B7D1"
BACKGROUND = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.1)"
TEXT_COLOR = "#E2E8F0"

CHART_LAYOUT = dict(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=BACKGROUND,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)


def regime_comparison_bar(old_tax: float, new_tax: float) -> go.Figure:
    """Side-by-side bar chart comparing old vs new regime tax."""
    fig = go.Figure(data=[
        go.Bar(
            name="Old Regime",
            x=["Tax Liability"],
            y=[old_tax],
            marker_color=OLD_REGIME_COLOR,
            text=[f"₹{old_tax:,.0f}"],
            textposition="outside",
            textfont=dict(size=14, color=TEXT_COLOR),
        ),
        go.Bar(
            name="New Regime",
            x=["Tax Liability"],
            y=[new_tax],
            marker_color=NEW_REGIME_COLOR,
            text=[f"₹{new_tax:,.0f}"],
            textposition="outside",
            textfont=dict(size=14, color=TEXT_COLOR),
        ),
    ])
    fig.update_layout(
        **CHART_LAYOUT,
        title="Old vs New Regime Tax Comparison",
        barmode="group",
        yaxis=dict(gridcolor=GRID_COLOR, title="Tax (₹)", tickformat=",.0f"),
        xaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


def deductions_donut(breakdown: Dict[str, float]) -> go.Figure:
    """Donut chart showing deduction breakdown."""
    labels = [k.replace("_", " ").title() for k, v in breakdown.items() if v > 0]
    values = [v for v in breakdown.values() if v > 0]

    if not values:
        return go.Figure()

    colors = px.colors.qualitative.Set2[:len(values)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#1A202C", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color=TEXT_COLOR),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
    )])

    total = sum(values)
    fig.add_annotation(
        text=f"₹{total:,.0f}<br>Total",
        x=0.5, y=0.5,
        font=dict(size=14, color=TEXT_COLOR),
        showarrow=False,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title="Deductions Breakdown",
        showlegend=True,
    )
    return fig


def tax_savings_gauge(savings: float, max_savings: float = 200000) -> go.Figure:
    """Gauge chart showing potential tax savings."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=savings,
        title={"text": "Potential Tax Savings", "font": {"color": TEXT_COLOR, "size": 16}},
        number={"prefix": "₹", "font": {"size": 24, "color": TEXT_COLOR}, "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, max_savings], "tickformat": ",.0f", "tickcolor": TEXT_COLOR},
            "bar": {"color": SAVINGS_COLOR},
            "bgcolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0, max_savings * 0.33], "color": "rgba(255,107,107,0.15)"},
                {"range": [max_savings * 0.33, max_savings * 0.66], "color": "rgba(255,193,7,0.15)"},
                {"range": [max_savings * 0.66, max_savings], "color": "rgba(78,205,196,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#FFD700", "width": 3},
                "thickness": 0.75,
                "value": savings,
            },
        },
    ))
    fig.update_layout(**CHART_LAYOUT, height=280)
    return fig


def slab_waterfall(regime: str, taxable_income: float, slabs: List[Dict]) -> go.Figure:
    """Waterfall chart showing how tax builds up across slabs."""
    names = [s["label"] for s in slabs]
    values = [s["tax"] for s in slabs]

    fig = go.Figure(go.Waterfall(
        name=f"{regime} Regime",
        orientation="v",
        measure=["relative"] * len(values) + ["total"],
        x=names + ["Total Tax"],
        y=values + [sum(values)],
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        decreasing={"marker": {"color": SAVINGS_COLOR}},
        increasing={"marker": {"color": OLD_REGIME_COLOR if regime == "Old" else NEW_REGIME_COLOR}},
        totals={"marker": {"color": "#FFD700"}},
        text=[f"₹{v:,.0f}" for v in values] + [f"₹{sum(values):,.0f}"],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{regime} Regime — Tax Build-up by Slab",
        yaxis=dict(gridcolor=GRID_COLOR, tickformat=",.0f"),
    )
    return fig


def optimization_horizontal_bar(opportunities: List[Dict]) -> go.Figure:
    """Horizontal bar chart for tax savings opportunities."""
    if not opportunities:
        return go.Figure()

    titles = [o["title"][:35] + "…" if len(o["title"]) > 35 else o["title"] for o in opportunities]
    savings = [o["estimated_tax_savings"] for o in opportunities]
    priorities = [o.get("priority", "MEDIUM") for o in opportunities]

    colors = [
        "#FF6B6B" if p == "HIGH" else "#FFD700" if p == "MEDIUM" else "#4ECDC4"
        for p in priorities
    ]

    fig = go.Figure(go.Bar(
        y=titles,
        x=savings,
        orientation="h",
        marker_color=colors,
        text=[f"₹{s:,.0f}" for s in savings],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
        hovertemplate="<b>%{y}</b><br>Savings: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Tax Saving Opportunities",
        xaxis=dict(gridcolor=GRID_COLOR, tickformat=",.0f", title="Estimated Savings (₹)"),
        yaxis=dict(autorange="reversed"),
        height=max(250, len(opportunities) * 60),
    )
    return fig


def opp_stacked_bar(opportunities):
    """
    Stacked horizontal bar: used investment vs remaining gap, per-bar colour by priority.
    """
    if not opportunities:
        return go.Figure()

    labels = [o["title"][:34] + "..." if len(o["title"]) > 34 else o["title"] for o in opportunities]
    used   = [o.get("current_investment", 0) for o in opportunities]
    gap    = [o.get("remaining_capacity", 0) for o in opportunities]
    pct    = [round(u / (u + g) * 100) if (u + g) > 0 else 0 for u, g in zip(used, gap)]

    pri_used_color = {"HIGH": "#FF6B6B", "MEDIUM": "#FFD700", "LOW": "#4ECDC4"}
    pri_gap_color  = {"HIGH": "rgba(255,107,107,0.18)", "MEDIUM": "rgba(255,215,0,0.15)", "LOW": "rgba(78,205,196,0.15)"}
    used_colors = [pri_used_color.get(o.get("priority","MEDIUM"), "#FF6B6B") for o in opportunities]
    gap_colors  = [pri_gap_color.get(o.get("priority","MEDIUM"), "rgba(255,255,255,0.08)") for o in opportunities]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Already Invested",
        y=labels, x=used, orientation="h",
        marker=dict(color=used_colors, opacity=0.92),
        text=[f"Rs.{v:,.0f} ({p}%)" if v > 0 else "" for v, p in zip(used, pct)],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#0f172a", size=11, family="Inter, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Invested: Rs.%{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Remaining Gap (Unused Limit)",
        y=labels, x=gap, orientation="h",
        marker=dict(color=gap_colors, line=dict(color=used_colors, width=1.5)),
        text=[f"Rs.{v:,.0f} unused" if v > 0 else "" for v in gap],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color=TEXT_COLOR, size=10, family="Inter, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Gap: Rs.%{x:,.0f}<extra></extra>",
    ))
    _layout = {k: v for k, v in CHART_LAYOUT.items() if k != "legend"}
    fig.update_layout(
        **_layout,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
        ),
        barmode="stack",
        title=dict(text="Deduction Utilisation  Used vs Remaining Limit", font=dict(size=15, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, tickformat=",.0f", title="Amount (Rs.)"),
        yaxis=dict(autorange="reversed"),
        height=max(320, len(opportunities) * 72),
    )
    return fig


def opp_treemap(opportunities):
    """
    Treemap: cell size = tax savings, rich gradient colours per priority.
    """
    if not opportunities:
        return go.Figure()

    # Richer colour palette per priority
    priority_colors = {
        "HIGH":   ["#FF6B6B", "#ff4444", "#e63946"],
        "MEDIUM": ["#FFD700", "#FFC300", "#f4a261"],
        "LOW":    ["#4ECDC4", "#38b2ac", "#2dd4bf"],
    }
    labels, parents, values, colors, custom = [], [], [], [], []
    for i, o in enumerate(opportunities):
        pri = o.get("priority", "MEDIUM")
        clr_list = priority_colors.get(pri, priority_colors["MEDIUM"])
        clr = clr_list[i % len(clr_list)]
        labels.append(o["title"])
        parents.append("Total Savings")
        values.append(max(o.get("estimated_tax_savings", 1), 1))
        colors.append(clr)
        custom.append(
            "Sec {}<br>Invest gap: Rs.{:,.0f}<br>Tax saved: Rs.{:,.0f}<br>Priority: {}".format(
                o.get("section", ""), o.get("remaining_capacity", 0),
                o.get("estimated_tax_savings", 0), pri
            )
        )

    labels  = ["Total Savings"] + labels
    parents = [""] + parents
    values  = [0] + values
    colors  = ["#1e293b"] + colors
    custom  = [""] + custom

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(width=2, color="#0f172a"),
                    pad=dict(t=6, b=6, l=6, r=6)),
        customdata=custom,
        hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
        texttemplate="<b>%{label}</b><br>Rs.%{value:,.0f}",
        textfont=dict(size=13, color="#f1f5f9", family="Inter, sans-serif"),
        textposition="middle center",
        branchvalues="total",
    ))
    _layout = {k: v for k, v in CHART_LAYOUT.items() if k not in ("margin",)}
    fig.update_layout(
        **_layout,
        title=dict(text="Tax Savings Map  block size = savings | RED=HIGH, AMBER=MEDIUM, TEAL=LOW", font=dict(size=14, color=TEXT_COLOR)),
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def savings_breakdown_donut(opportunities):
    """
    Colourful donut: each slice = one deduction opportunity.
    Legend shows colour + section name + Rs. saved.
    Centre annotation shows total.
    """
    if not opportunities:
        return go.Figure()

    VIBRANT = [
        "#FF6B6B", "#FFD700", "#4ECDC4", "#A78BFA",
        "#FB923C", "#34D399", "#60A5FA", "#F472B6",
        "#FBBF24", "#10B981",
    ]

    rich_labels, values, colors, custom = [], [], [], []
    for i, o in enumerate(opportunities):
        sv = o.get("estimated_tax_savings", 0)
        if sv <= 0:
            continue
        clr = VIBRANT[i % len(VIBRANT)]
        sec = str(o.get("section", ""))
        title = o["title"][:26]
        rich_labels.append("Sec " + sec + "  Rs." + "{:,.0f}".format(sv))
        values.append(sv)
        colors.append(clr)
        custom.append(
            "<b>" + title + "</b><br>Section " + sec +
            "<br>Tax Saving: Rs." + "{:,.0f}".format(sv) +
            "<br>Priority: " + o.get("priority", "MEDIUM")
        )

    if not values:
        return go.Figure()

    total = sum(values)
    pct_labels = [
        lbl + "  (" + str(int(v / total * 100)) + "%)"
        for lbl, v in zip(rich_labels, values)
    ]

    fig = go.Figure(data=[go.Pie(
        labels=pct_labels,
        values=values,
        hole=0.60,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="none",
        customdata=custom,
        hovertemplate="%{customdata}<extra></extra>",
        pull=[0.06 if v == max(values) else 0 for v in values],
        rotation=45,
        showlegend=True,
    )])
    fig.add_annotation(
        text="<b>Rs." + "{:,.0f}".format(total) + "</b><br>Total Savings",
        x=0.5, y=0.5,
        font=dict(size=16, color="#FFD700", family="Inter, sans-serif"),
        showarrow=False,
        align="center",
    )
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        title=dict(
            text="Where do the savings come from?",
            font=dict(size=16, color=TEXT_COLOR, family="Inter, sans-serif"),
        ),
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(size=12, color=TEXT_COLOR),
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            itemsizing="constant",
        ),
        showlegend=True,
        height=420,
        margin=dict(l=20, r=240, t=60, b=20),
    )
    return fig