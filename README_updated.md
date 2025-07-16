# Quantitative Risk & Fixed Income Projects

This repository showcases a suite of quantitative finance projects focused on multi-asset risk analysis, non-parametric fixed-income risk modeling, and bond pricing via simulation and factor modeling. These implementations reflect practical quantitative techniques applicable to risk management, fixed income analytics, and quantitative research.

---

## 📌 Project 1: Multi-Asset Portfolio Risk Analysis

**Objective:**  
Construct and analyze three equity-based portfolios (Value, Growth, Industrial ETFs) using 10 distinct Value-at-Risk (VaR) methodologies and evaluate their statistical properties and coherence.

**Key Features:**
- Portfolios: 10-stock equal-weighted compositions based on investment styles.
- Risk Models:
  - Parametric Normal VaR
  - Historical Simulation
  - Monte Carlo Simulation
  - GARCH-based VaR
  - Filtered Historical Simulation
  - Extreme Value Theory (EVT)
  - Cornish-Fisher Expansion
  - Kernel Density Estimation
  - Bootstrapped VaR
  - Student’s t-distribution VaR
- Additional Metrics:
  - Expected Shortfall (ES)
  - Sharpe and Sortino Ratios
  - Maximum Drawdown, Conditional Drawdown at Risk (CDaR)
- Statistical Testing:
  - Normality (Shapiro-Wilk, KS Test)
  - Stationarity and Autocorrelation
- Coherent Risk Measure Validation (Monotonicity, Sub-additivity, etc.)

📈 *Deliverables include documented Python notebooks, statistical diagnostics, and performance visualizations.*

---

## 📌 Project 2: Python Library for Non-Parametric Risk in Fixed-Income

**Objective:**  
Develop a Python module for estimating non-parametric risk measures—VaR, ES, EVT—for both single bonds and fixed-income portfolios without assuming return distributions.

**Key Components:**
- `FixedIncomeNprmSingle` class:
  - Quantile- and bootstrap-based VaR/ES.
  - EVT estimation using Generalized Pareto Distribution (GPD).
- `FixedIncomeNprmPort` class:
  - Aggregates multi-asset fixed-income portfolios.
  - Marginal VaR per position using sensitivity approximations.
- Modular Design:
  - Input validation functions (`inputsControlFunctions.py`)
  - Custom loss functions and EVT optimizers (`SupportFunctions.py`)
- Fully tested on simulated and historical bond return data.

🔧 *Designed for portfolio risk managers seeking robust, distribution-free methods in fixed income.*

---

## 📌 Project 3: Interest Rate Risk Assessment and Bond Pricing

**Objective:**
Build a robust framework for pricing fixed-income securities and assessing their exposure to interest rate risk using both analytical and simulation-based methods.

**Core Methods:**
	•	Bond pricing for both coupon and zero-coupon instruments using interpolated yield curves
	•	Yield-to-Maturity (YTM) solved via Newton’s method
	•	Duration metrics: Macaulay and Modified Duration
	•	Value-at-Risk (VaR) & Expected Shortfall (ES) via:
	•	Modified Duration Mapping with volatility estimation (EWMA, GARCH, Simple)
	•	Historical Simulation with spline-interpolated yield curve shocks
	•	Risk decomposition across time decay and yield movement

🧠 This project highlights essential techniques for fixed-income valuation, duration-based risk mapping, and historical stress testing of bond portfolios.

⸻

**🧩 Technical Stack**
	•	Python 3.10+
	•	Libraries: numpy, pandas, matplotlib, scipy, arch, statsmodels
	•	Techniques: Convexity-adjusted analytics, time-series modeling, spline interpolation
	•	Tools: Jupyter Notebooks, Git version control

⸻

**🔍 Why This Matters**

This work bridges theory with practice in bond pricing and fixed-income risk modeling. It’s especially relevant for:
	•	Fixed-Income Trading & Quantitative Strategy
	•	Risk Management (IRRBB, VaR, Stress Testing)
	•	Sell-Side Quantitative Research & Credit Structuring
	•	Portfolio Risk Analytics for Institutional Investors

📬 *Reach out on LinkedIn for collaboration or conversation in quantitative finance.*

---

## 📂 Repository Structure

```
/PortfolioRiskAnalysis/             ← Project 1: Equity VaR and Risk Evaluation
/FixedIncomeNonParametricLibrary/  ← Project 2: Non-Parametric Fixed Income Risk
/BondPricingSimulation/            ← Project 3: Bond Pricing & Factor Models
README.md
```

---

## 👨‍💻 Author

**Runbo Ye**  
Senior @ University of Michigan | Majoring in Financial Mathematics & Data Science  
Focused on Quantitative Finance, Risk Modeling, and Fixed Income Analytics  
📫 [LinkedIn](https://www.linkedin.com/in/runboye/) | 🌐 GitHub: `Rambo-code668`
