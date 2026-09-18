# NEPALI Pretraining Data Cleaning & Token Budget Report

This report tracks the document, word, and token progression across all pipeline stages.

| Stage | Manual Docs | Manual Words | Manual Tokens | Downloaded Docs | Downloaded Words | Downloaded Tokens | Total Tokens | Manual % | Status |
|---|---|---|---|---|---|---|---|---|---|
| Final Deduplicated & Filtered | 144,332 | 63,108,277 | 126,573,349 | 4,047,379 | 609,395,558 | 1,245,462,400 | 1,372,035,749 | 9.2% | ⚠️ LOW |

---

### 🎯 Token Budget Guard Criteria:
- **Manual Training Tokens**: $\ge 100\text{M}$ tokens ($\ge 20\%$ of target corpus).
- **Downloaded Training Tokens**: $\approx 370\text{M}$ tokens (after heavy PPL & foreign word filter).
- **Total Training Budget**: $\approx 500\text{M}$ tokens.