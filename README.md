# TradeLENS

TradeLENS is a public research repository for stock recommendation analysis, explainability, and visualization.

## Repository overview

- `Dashboard/` - Vue 3 + Quasar dashboard for browsing asset recommendation signals and mock recommendation output.
- `Data/` - Analysis outputs, financial performance summaries, SHAP explainability tables, and result manifests.
- `Final_Manuscript/` - Final report assets, figures, and LaTeX source files.
- `Luceres_Proposal_Manuscript/` - Proposal documentation and manuscript source materials.

## Dashboard

The dashboard is built with Vite, Vue 3, and Quasar.

### Run locally

```bash
cd Dashboard
npm install
npm run dev
```

Then open the local dev server URL shown in the terminal.

### Notes

- The dashboard uses a local mock dataset for recommendations.
- Recommendation outputs are displayed as `Buy`, `Hold`, or `Sell`.
- Recommendation colors in the UI:
  - `Buy` = green
  - `Hold` = yellow
  - `Sell` = red

## Usage

- Use the `Dashboard/` folder for interactive visual exploration.
- Use the `Data/` folder for model outputs, SHAP summaries, and performance files.
- Use the manuscript folders for report writing and documentation.
