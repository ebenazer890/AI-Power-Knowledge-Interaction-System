from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FinanceParseResult:
    df: pd.DataFrame
    datetime_col: str
    amount_col: str
    category_col: Optional[str]


_AMOUNT_RE = re.compile(r"[-+]?\$?\s*\(?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?")
_DATE_HINTS = ("date", "time", "datetime", "timestamp")
_AMOUNT_HINTS = ("amount", "amt", "value", "total", "debit", "credit", "payment", "balance")
_CATEGORY_HINTS = ("category", "type", "merchant", "description", "desc", "account")


def _normalize_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def _to_number(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # detect parentheses as negative
    neg = "(" in s and ")" in s
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"[^0-9.\-+]", "", s)
    if not s:
        return None
    try:
        v = float(s)
        return -abs(v) if neg else v
    except Exception:
        return None


def _best_col(cols: List[str], hints: Tuple[str, ...]) -> Optional[str]:
    norm = {c: _normalize_col(c) for c in cols}
    for h in hints:
        for c, nc in norm.items():
            if h in nc:
                return c
    return None


def _detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    hinted = _best_col(cols, _AMOUNT_HINTS)
    if hinted:
        return hinted

    # Otherwise: choose col with most amount-like values
    best = None
    best_score = 0
    for c in cols:
        series = df[c].dropna().astype(str)
        if series.empty:
            continue
        hits = series.str.contains(_AMOUNT_RE).mean()
        if hits > best_score:
            best_score = float(hits)
            best = c
    return best if best_score >= 0.3 else None


def _detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    hinted = _best_col(cols, _DATE_HINTS)
    if hinted:
        return hinted

    # Otherwise: try parse each column and take the best success rate
    best = None
    best_rate = 0.0
    for c in cols:
        s = df[c]
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best_rate = rate
            best = c
    return best if best_rate >= 0.3 else None


def _detect_category_col(df: pd.DataFrame, *, exclude: List[str]) -> Optional[str]:
    cols = [c for c in df.columns if c not in exclude]
    hinted = _best_col(cols, _CATEGORY_HINTS)
    if hinted:
        return hinted

    # choose a low-cardinality text column
    best = None
    best_score = 0.0
    for c in cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        unique = s.nunique()
        n = len(s)
        # prefer moderate uniqueness
        ratio = unique / max(1, n)
        score = 1.0 - abs(ratio - 0.2)
        if score > best_score:
            best_score = score
            best = c
    return best


def parse_financial_tables(tables: List[List[object]]) -> Optional[FinanceParseResult]:
    """Heuristic parser: takes a list of table rows (each row a list of cells).

    Expects first row to be headers for at least one table.
    Returns the best candidate table as a DataFrame with detected datetime/amount/category columns.
    """
    best: Optional[FinanceParseResult] = None
    best_rows = 0

    for t in tables:
        if not t or len(t) < 2:
            continue

        header = [str(x or "").strip() for x in t[0]]
        if sum(1 for h in header if h) < 2:
            continue

        body = t[1:]
        df = pd.DataFrame(body, columns=header)
        df = df.dropna(how="all")
        if len(df) < 3:
            continue

        dt_col = _detect_datetime_col(df)
        amt_col = _detect_amount_col(df)
        if not dt_col or not amt_col:
            continue

        df2 = df.copy()
        df2[dt_col] = pd.to_datetime(df2[dt_col], errors="coerce")
        df2[amt_col] = df2[amt_col].map(_to_number)
        df2 = df2.dropna(subset=[dt_col, amt_col])
        if len(df2) < 3:
            continue

        cat_col = _detect_category_col(df2, exclude=[dt_col, amt_col])

        candidate = FinanceParseResult(df=df2, datetime_col=dt_col, amount_col=amt_col, category_col=cat_col)
        if len(df2) > best_rows:
            best_rows = len(df2)
            best = candidate

    return best


def aggregate_finance(
    parsed: FinanceParseResult,
    *,
    freq: str,
) -> pd.DataFrame:
    """Aggregate by time freq. freq in {'H','D','M'}"""
    df = parsed.df.copy()
    dt = parsed.datetime_col
    amt = parsed.amount_col

    df = df.sort_values(dt)
    df = df.set_index(dt)
    agg = df[amt].resample(freq).sum().to_frame(name="total_amount")
    agg["abs_total"] = agg["total_amount"].abs()
    return agg.reset_index().rename(columns={dt: "period"})


def totals(parsed: FinanceParseResult) -> dict:
    amt = parsed.amount_col
    s = parsed.df[amt].astype(float)
    return {
        "rows": int(len(parsed.df)),
        "sum": float(s.sum()),
        "income_sum_pos": float(s[s > 0].sum()),
        "expense_sum_neg": float(s[s < 0].sum()),
        "mean": float(s.mean()),
    }


def pie_breakdown(parsed: FinanceParseResult, *, top_n: int = 8) -> Tuple[pd.DataFrame, str]:
    """Return (df, label) where df has columns label/value for a pie chart."""
    amt = parsed.amount_col
    df = parsed.df.copy()

    if parsed.category_col:
        cat = parsed.category_col
        grp = df.groupby(cat, dropna=True)[amt].sum().abs().sort_values(ascending=False)
        grp = grp.head(top_n)
        out = grp.reset_index().rename(columns={cat: "label", amt: "value"})
        return out, "By category (absolute amount)"

    # fallback: by month
    dt = parsed.datetime_col
    df["month"] = pd.to_datetime(df[dt]).dt.to_period("M").astype(str)
    grp = df.groupby("month")[amt].sum().abs().sort_values(ascending=False)
    grp = grp.head(top_n)
    out = grp.reset_index().rename(columns={"month": "label", amt: "value"})
    return out, "By month (absolute amount)"


def advanced_summary(parsed: FinanceParseResult) -> Dict[str, float]:
    """More detailed numeric summary for the detected financial table."""
    amt = parsed.amount_col
    s = parsed.df[amt].astype(float)

    pos = s[s > 0]
    neg = s[s < 0]
    return {
        "rows": float(len(s)),
        "net_sum": float(s.sum()),
        "abs_sum": float(s.abs().sum()),
        "income_sum_pos": float(pos.sum()) if len(pos) else 0.0,
        "expense_sum_neg": float(neg.sum()) if len(neg) else 0.0,
        "mean": float(s.mean()) if len(s) else 0.0,
        "median": float(s.median()) if len(s) else 0.0,
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()) if len(s) else 0.0,
        "max": float(s.max()) if len(s) else 0.0,
        "income_count": float(len(pos)),
        "expense_count": float(len(neg)),
    }


def top_transactions(parsed: FinanceParseResult, *, n: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_income, top_expense) by absolute amount."""
    df = parsed.df.copy()
    dt = parsed.datetime_col
    amt = parsed.amount_col

    cols = [dt, amt]
    if parsed.category_col and parsed.category_col in df.columns:
        cols.append(parsed.category_col)

    base = df[cols].copy().sort_values(dt)
    income = base[base[amt] > 0].sort_values(amt, ascending=False).head(n)
    expense = base[base[amt] < 0].sort_values(amt, ascending=True).head(n)
    return income.reset_index(drop=True), expense.reset_index(drop=True)


def category_breakdown(parsed: FinanceParseResult, *, top_n: int = 12) -> Tuple[pd.DataFrame, str]:
    """Return (df, label) where df has category + value for bar charts."""
    if not parsed.category_col:
        return pd.DataFrame(), ""
    cat = parsed.category_col
    amt = parsed.amount_col

    grp = parsed.df.groupby(cat, dropna=True)[amt].sum().abs().sort_values(ascending=False).head(top_n)
    out = grp.reset_index().rename(columns={cat: "label", amt: "value"})
    return out, "Top categories (absolute amount)"


def parse_finance_request(user_text: str) -> Dict[str, str]:
    """Tiny rule-based intent parser for finance UI.

    Returns keys like: {"chart": "bar|pie|line|none", "group": "category|hour|day|month|auto"}
    """
    t = " ".join((user_text or "").strip().lower().split())
    chart = "none"
    if any(k in t for k in ("bar chart", "barchart", "bar")):
        chart = "bar"
    elif any(k in t for k in ("pie chart", "pie")):
        chart = "pie"
    elif any(k in t for k in ("line chart", "trend", "over time", "time series", "timeline", "line")):
        chart = "line"

    group = "auto"
    if any(k in t for k in ("category", "merchant", "type", "by category")):
        group = "category"
    elif any(k in t for k in ("hour", "hourly")):
        group = "hour"
    elif any(k in t for k in ("day", "daily")):
        group = "day"
    elif any(k in t for k in ("month", "monthly")):
        group = "month"

    return {"chart": chart, "group": group}
