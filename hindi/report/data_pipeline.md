# HINDI Pretraining Data Cleaning & Token Budget Report

This report tracks the document, word, and token progression across all pipeline stages.

| Stage | Manual Docs | Manual Words | Manual Tokens | Downloaded Docs | Downloaded Words | Downloaded Tokens | Total Tokens | Manual % | Status |
|---|---|---|---|---|---|---|---|---|---|
| Final Deduplicated & Filtered | 125,049 | 55,215,574 | 89,765,392 | 1,081,394 | 90,396,743 | 143,673,376 | 233,438,768 | 38.5% | ⚠️ LOW |

---

### 🎯 Token Budget Guard Criteria:
- **Manual Training Tokens**: $\ge 100\text{M}$ tokens ($\ge 20\%$ of target corpus).
- **Downloaded Training Tokens**: $\approx 370\text{M}$ tokens (after heavy PPL & foreign word filter).
- **Total Training Budget**: $\approx 500\text{M}$ tokens.