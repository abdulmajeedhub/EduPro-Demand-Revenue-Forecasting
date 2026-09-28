# EduPro – Predictive Modeling for Course Demand & Revenue Forecasting

## Full Package Deliverables

| File | Description |
|------|-------------|
| `EduPro_Research_Paper.pdf` | Complete research paper (EDA, methodology, model results, insights, recommendations) |
| `EduPro_Executive_Summary.pdf` | One-page executive summary for stakeholders |
| `app.py` | Streamlit dashboard (3-colour theory only) |
| `Courses.csv` / `Teachers.csv` / `Transactions.csv` | Synthetic datasets matching project schema |
| `courses_processed.csv` | Feature-engineered dataset used by models |
| `model_enrollment.pkl` / `model_revenue.pkl` | Trained Gradient Boosting models |
| `feature_importance_*.csv` | Feature importance rankings |
| `category_stats.csv` | Category-level aggregates |

## Colour Palette (Colour Theory – Analogous Cool)

- **Primary** `#0D47A1` – Deep Professional Blue (trust, headers, primary actions)
- **Secondary** `#00897B` – Soft Teal (accents, charts, success states)
- **Neutral** `#F5F7FA` – Cool Off-White (backgrounds, cards)

Chosen for an education platform: high readability, professional trust, calm modern feel.

## How to Run the Streamlit Dashboard

```bash
streamlit run app.py
```

## Dashboard Modules

1. **Overview Dashboard** – KPIs + category enrollment/revenue charts + model performance snapshot
2. **Historical Insights** – Price sensitivity, rating impact, level & duration effects
3. **Feature Importance** – Ranked drivers for enrollment & revenue + business translation table
4. **Interactive Predictor** – Live what-if tool (price, duration, level, ratings, experience → predicted enrollments & revenue)
5. **Category Comparison** – Side-by-side category performance
6. **About & Methodology** – Full documentation of the approach

## Model Summary

- Best model: **Gradient Boosting**
- Enrollment R² ≈ 0.78 | Revenue R² ≈ 0.81
- Top enrollment driver: Course Rating
- Top revenue driver: Course Price
