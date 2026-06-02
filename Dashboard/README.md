# Quasar Stock Recommendation Dashboard

Simple Quasar Framework dashboard built with Vue 3 Composition API. It uses a local mock dataset to return a `Buy`, `Hold`, or `Sell` recommendation for a selected asset and date.

## Folder

Project path:

`quasar-stock-dashboard`

## Run

1. Open a terminal in `quasar-stock-dashboard`
2. Install dependencies:

```bash
npm install
```

3. Start the dev server:

```bash
npm run dev
```

## Notes

- Mock data is stored in `src/data/mockRecommendations.js`
- Main UI is in `src/App.vue`
- Recommendation colors:
  - `Buy` = green
  - `Hold` = yellow
  - `Sell` = red
