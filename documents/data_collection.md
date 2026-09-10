# Chapter 1: The Data

## Building the Foundation
Every machine learning project begins with data. For gold price prediction, we needed more than just historical gold prices, we needed a comprehensive view of the economic landscape. We collected 17 features spanning 20+ years (2000-2025) from four major categories: **Precious Metals**, **Financial Markets**, **Macroeconomics** and **Geopolitical Risk**. This rich dataset forms the foundation of our analysis.

## Feature Categories (17 Total)

### Precious Metals (3 features)
- **Gold Spot:** This is our Target Variable. It represents the price of physical gold (per ounce) in the global market. Our goal is to predict this value.
- **Gold Futures (GC=F):** Futures contracts reflect market expectations of where gold prices will be at a future date. It's a key indicator of market sentiment.
- **Silver Futures (SI=F):** Silver is a precious metal highly correlated with gold. It often moves in the same direction, representing co-movement in the metals market.

### Financial Markets (5 features)
- **S&P 500 & NASDAQ (^GSPC, ^IXIC):** Major U.S. stock indices. They are often inversely correlated with gold. When stocks are up ('risk-on'), investors may sell gold (a 'safe-haven' asset).
- **Crude Oil (CL=F):** Oil prices heavily influence inflation (which gold is a hedge against) and the overall cost of industrial production.
- **VIX Index (^VIX):** The 'Fear Index'. It measures expected market volatility. When VIX is high (high fear), demand for gold as a safe-haven asset typically increases.
- **Gold ETF (GLD):** SPDR Gold Shares (GLD) is an Exchange-Traded Fund. Its holdings reflect direct financial investment demand for gold from retail and institutional investors.

### Macroeconomic (6 features)
- **CPI (Inflation) (CPIAUCSL):** Consumer Price Index. This is a key measure of inflation. Gold is traditionally seen as a hedge against inflation, so as CPI rises, gold demand often follows.
- **Fed Funds Rate (FEDFUNDS):** The benchmark interest rate. Higher rates make interest-bearing assets (like bonds) more attractive, reducing the appeal of gold (which pays no interest).
- **10Y Treasury & Real Rate (GS10, DFII10):** The real interest rate (Treasury yield minus inflation). This is a critical driver. When real rates are low or negative, the 'opportunity cost' of holding gold is low, making it more attractive.
- **USD Index (DTWEXBGS):** Measures the strength of the U.S. Dollar. Since gold is priced in USD, a stronger dollar makes gold more expensive for foreign buyers, often lowering demand and price (and vice-versa).
- **M2 Money Supply (M2SL):** Represents the total amount of money in the economy. A rapid increase in money supply can lead to inflation fears, boosting gold's appeal.
- **Unemployment Rate (UNRATE):** A key indicator of economic health. High unemployment can signal economic distress, increasing demand for gold as a safe-haven asset.

### Geopolitical Risk (3 features)
- **GPR (Risk Index) (GPR):** The Geopolitical Risk Index (GPR) measures tensions from news reports. High geopolitical risk (wars, conflicts) drives investors to safe-haven assets like gold.
- **GPR Acts (GPRA):** A subset of the GPR index that measures only concrete geopolitical 'acts' (e.g., a new conflict starting).
- **GPR Threats (GPRT):** A subset of the GPR index that measures geopolitical 'threats' (e.g., new war threats).

## Why These Categories?
Gold prices don't exist in isolation, they are shaped by a complex interplay of economic, financial, and geopolitical forces. Our feature selection is grounded in economic theory and empirical research.
- **Inflation** (measured through CPI and Real Interest Rates) directly affects gold's role as a store of value. When inflation rises, investors flock to gold to preserve purchasing power.
- **Market Sentiment** (captured by stock indices, VIX, and commodity prices) reflects investor risk appetite during 'risk-on' periods, capital flows to equities; during 'risk-off' periods, it shifts to safe havens like gold.
- **Monetary Policy** (Federal Funds Rate, M2 Money Supply, and Treasury Yields) influences the opportunity cost of holding non-yielding assets like gold. Finally,
- **Geopolitical Risk** (GPR indices) measures global uncertainty and conflict, which historically drives demand for gold as a crisis hedge.

These 17 features are not arbitrary—they represent the fundamental drivers that economics research has identified as key determinants of gold prices over the past two decades.

## Our Data Sources
- **Yahoo Finance (yfinance):** Real-time and historical market data for commodities, indices, and ETFs. (Frequency: Daily)
- **FRED API:** Federal Reserve Economic Data - comprehensive macroeconomic indicators. (Frequency: Monthly)
- **World Gold Council (WGC):** Official source for our target variable: the daily Gold Spot Price. (Frequency: Monthly)
- **GPR Database:** Measures global geopolitical tensions by Caldara & Iacoviello (2022). (Frequency: Monthly)

## The Foundation is Set
With our raw data collected, we have the building blocks for our models. However, this data is messy: it has different frequencies (Daily vs. Monthly) and missing values. In the next chapter, we will clean, process, and explore this data to uncover its hidden stories.