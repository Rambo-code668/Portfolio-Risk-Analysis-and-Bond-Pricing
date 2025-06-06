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

## 📌 Project 3: Bond Pricing via Simulation and Factor Modeling

**Objective:**  
Implement and compare various bond pricing models using Monte Carlo simulations, interest rate factor models, and credit spread modeling.

**Techniques Covered:**
- Short-rate models and term structure simulation
- Bond price sensitivity to macroeconomic and yield curve factors
- Yield-to-Maturity (YTM), Duration, and Convexity analysis
- Credit risk adjustment using empirical spreads
- Risk-neutral pricing of defaultable bonds
- Visualization of yield curves and forward rates

🧠 *This module emphasizes dynamic pricing of fixed-income securities and builds intuition for both credit and interest rate risk.*

---

## 🧩 Technical Stack

- Python 3.10+
- Libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `yfinance`, `arch`
- Tools: Jupyter Notebooks, Object-Oriented Programming, Git

---

## 🔍 Why This Matters

This repository demonstrates hands-on proficiency in risk quantification, fixed-income modeling, and simulation techniques. Skills are applicable to:

- Quantitative Risk & Research Roles (Buy-side / Sell-side)
- Fixed Income Strategy & Analytics
- Credit Portfolio Risk Management
- Structured Product Pricing

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
📫 [LinkedIn](https://www.linkedin.com/in/runbo-ye) | 🌐 GitHub: `Rambo-code668`