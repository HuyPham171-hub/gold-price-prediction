# Chapter 2: Exploratory Data Analysis - Understanding the Gold Price Landscape

## Executive Summary
### What We Discovered
- **Strong Inflation Link:** CPI explains 75% of gold price variance (r=0.87) – confirming gold's role as inflation hedge.
- **Inverse Rate Relationship:** Higher real interest rates lead to lower gold prices (r=-0.26) due to opportunity cost.
- **Equity Market Surprise:** Gold and S&P 500 move together (r=0.82), challenging the 'safe haven' narrative.
- **VIX Paradox:** Market volatility shows NO correlation with gold (r≈0.00) – unexpected finding!

In this chapter, we'll explore 17 features across 19.5 years (2006-2025) to understand the economic forces that drive gold prices. Through correlation analysis, distribution studies, and interactive visualizations, we'll identify the 13 most predictive features for our models.

## Data Collection Journey: Finding Common Ground
Imagine trying to combine 7 different historical records, each starting at different points in time. Our first challenge was finding when ALL these records overlap - the moment where every piece of data becomes available. We use Gold Spot Price as our foundation (measured monthly), and match all other indicators to its timeline. This ensures our analysis uses consistent monthly snapshots from start to finish.

### Data Availability by Source
| Dataset | Earliest Valid Date | Latest Date | Original Frequency |
|---------|---------------------|-------------|--------------------|
| Market Data (S&P 500, NASDAQ, etc.) | 2004-11-18 | 2025-05-30 | Daily |
| **USD Index** | **2006-01-02** | 2025-05-30 | Daily |
| Macroeconomic (CPI, Unemployment, M2) | 2000-01-01 | 2025-05-01 | Monthly |
| Real Interest Rate | 2003-01-02 | 2025-05-01 | Monthly |
| VIX (Volatility Index) | 2000-01-03 | 2025-05-30 | Daily |
| GPR (Geopolitical Risk) | 1985-01-01 | 2025-05-01 | Monthly |
| Gold Spot Price (WGC) | 1978-01-31 | 2025-05-31 | Monthly |

**Common Start Date: January 31, 2006**
This is the first month-end after USD Index becomes available (Jan 2, 2006). Since we join all data to Gold Spot Price (monthly), the timeline starts at the nearest month-end. From this date forward, we have 19.5 years (233 months) of complete data across all features.

### How We Combined the Data
1. **Step 1:** Find the 'starting line', when the last dataset begins (USD Index on January 2, 2006)
2. **Step 2:** Use Gold Spot Price as the anchor, it's measured monthly (end of each month)
3. **Step 3:** Match all other data to these monthly dates, even if they're originally tracked daily
4. **Step 4:** Fill any small gaps, ensuring we have complete information for every month
5. **Step 5:** Final result, one unified monthly timeline from 2006 to 2025

## Dataset Overview
After alignment and preprocessing, we have a unified monthly dataset spanning 19.5 years. All 17 features are synchronized to end-of-month dates.
- Total Features: 17
- Time Span: 19.5 Years
- Monthly Observations: 233

## Why USD for Gold Spot? Understanding the Target Variable
Gold prices are quoted in multiple currencies globally (USD, EUR, GBP, JPY, CHF, etc.). Choosing the right currency for our target variable is crucial. We analyzed correlations across all available currencies and selected USD for three compelling reasons:

- **Near-Perfect Correlation:** Correlation matrix shows r > 0.99 between USD, EUR, GBP gold prices. The differences are purely exchange rate scaling, the underlying gold movement is identical. Some currencies show slightly lower correlations (0.74-0.79) due to shorter historical data.
- **Global Standard:** USD is the international pricing standard. Major gold exchanges (COMEX, LBMA) quote in USD/oz. Most economic indicators (CPI, Fed rates, S&P 500) are USD-based, ensuring consistency. Using USD eliminates currency conversion noise in our analysis.
- **Most Complete Data:** USD gold prices have the longest uninterrupted time series (1978-2025 from WGC). Other currencies have gaps or shorter histories. Gold_Spot_USD provides 47 years of data, though we only use 19.5 years (2006-2025) due to USD Index limitations.

### Gold Spot vs Gold Futures vs Gold ETF
Beyond currency selection, we also chose between three gold price representations. All three have correlation near 1.0, but differ in what they represent:

| Product | What It Represents | Market Characteristics | Decision |
|---------|--------------------|------------------------|----------|
| **Gold Spot** | Physical gold price for immediate delivery on OTC markets | Reflects real supply/demand of physical gold; less speculative | **SELECTED** |
| **Gold Futures** | Contracts for future delivery on COMEX exchange | High liquidity, many speculators; more volatile short-term | Rejected |
| **Gold ETF (GLD, IAU)** | Exchange-traded fund tracking gold; traded like stocks | Includes fund management fees; affected by equity flows | Rejected |

**Why Gold Spot Selected:**
- Represents actual physical gold value without speculative noise
- Long-term stability better reflects gold's role as store of value
- No contract expiration dates or rollover costs (unlike futures)
- No management fees or tracking errors (unlike ETFs)
- Best proxy for 'true' gold price in economic analyses

## Target Variable: Gold Spot Price
Gold spot price represents the current market price for immediate physical delivery. As our target variable, this time series (2006-2025) reveals long-term trends and volatility patterns. Gold grew from ~$600/oz in 2006 to over $2,700/oz in 2024, driven by financial crises, inflation fears, and geopolitical tensions.

- **2008 Financial Crisis:** Gold surged from $800 to $1,900 as investors fled to safe havens during the Great Recession
- **COVID-19 Pandemic (2020):** Gold hit record high of $2,067 amid unprecedented monetary stimulus and economic uncertainty
- **2024 All-Time High:** Gold broke $2,700/oz driven by inflation fears, Fed rate cuts, and geopolitical tensions

## Feature Distributions
Understanding value distributions helps detect outliers, skewness, and identify potential transformations needed for modeling. 
- **Target Variable:** Gold Spot price distribution shows right skewness with increasing trend over time. The boxplot reveals several outliers during crisis periods (2008, 2020, 2024).
- **Market Indicators:** Market indicators (S&P 500, USD Index, Silver, Crude Oil) show varying distribution patterns. Stock indices show strong upward trends, while commodities exhibit higher volatility.
- **Macroeconomic:** Macroeconomic variables (CPI, Unemployment, Interest Rates) reflect major policy shifts. CPI shows steady inflation growth, while rates fluctuated dramatically during QE periods.
- **Volatility & Risk:** Volatility indicators (VIX, GPR, GPRA) capture market fear and geopolitical tensions. VIX spikes during crises (2008, 2020), while GPR shows elevated levels during conflicts.

## Correlation & Relationships
Understanding feature relationships reveals which variables move together and helps identify multicollinearity. Red cells indicate strong positive correlations (variables rising together), while blue shows negative correlations (inverse relationships).

### Top Correlations with Gold Spot
| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| **CPI (Inflation)** | +0.85 | Strong positive – inflation drives gold demand |
| **M2 Money Supply** | +0.82 | Strong positive – monetary expansion boosts gold |
| **S&P 500** | +0.80 | Surprising! Both rise in liquidity-driven markets |
| **Silver Futures** | +0.71 | Strong positive – precious metals move together |
| **Real Interest Rate** | -0.40 | Negative – higher rates reduce gold appeal |
| **Treasury Yield 10Y** | -0.34 | Negative – bonds compete with gold |