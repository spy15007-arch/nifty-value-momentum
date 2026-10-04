import math
import yfinance as yf


def safe(value):

    try:

        if value is None:
            return None

        if math.isnan(float(value)):
            return None

        return float(value)

    except Exception:
        return None


def get_fundamentals(ticker):

    ticker_obj = yf.Ticker(ticker)

    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    def get(*names):

        for name in names:

            if info.get(name) is not None:
                return safe(info.get(name))

        return None

    market_cap = get("marketCap")

    pe = get("trailingPE")

    pb = get("priceToBook")

    ps = get("priceToSalesTrailing12Months")

    ev_ebitda = get("enterpriseToEbitda")

    roe = get("returnOnEquity")

    roce = get("returnOnCapitalEmployed")

    debt_equity = get("debtToEquity")

    eps = get("trailingEps")

    eps_growth = get("earningsGrowth")

    revenue_growth = get("revenueGrowth")

    fcf = None

    try:

        cashflow = ticker_obj.cashflow

        if cashflow is not None and not cashflow.empty:

            for name in [
                "Free Cash Flow",
                "FreeCashFlow",
            ]:

                if name in cashflow.index:

                    fcf = safe(
                        cashflow.loc[name].iloc[0]
                    )

                    break

    except Exception:
        pass

    fcf_yield = None

    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap

    earnings_yield = None

    if pe and pe > 0:
        earnings_yield = 1 / pe

    # yfinance may return D/E as a percentage-like value.
    if debt_equity is not None and debt_equity > 5:
        debt_equity = debt_equity / 100

    return {

        "market_cap": market_cap,

        "pe": pe,

        "pb": pb,

        "ps": ps,

        "ev_ebitda": ev_ebitda,

        "roe": roe,

        "roce": roce,

        "debt_equity": debt_equity,

        "eps": eps,

        "eps_growth": eps_growth,

        "revenue_growth": revenue_growth,

        "fcf_yield": fcf_yield,

        "earnings_yield": earnings_yield,
    }
