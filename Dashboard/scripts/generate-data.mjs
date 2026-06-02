import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const stocksRoot = path.resolve(projectRoot, '..', 'Thesis', 'Stocks')
const outputDir = path.resolve(projectRoot, 'src', 'data')
const outputPath = path.resolve(outputDir, 'generatedRecommendations.js')

const PREDICTION_FILE = path.join('tech_macro', 'timeCV_2020', 'test_predictions.csv')
const CATEGORY_LABELS = {
  'Blue chip stocks': 'Blue Chip Stocks',
  Crypto: 'Crypto',
  PennyStocks: 'Penny Stocks'
}
const ASSET_FULL_NAMES = {
  AC: 'Ayala Corporation',
  ARA: 'Araneta Properties, Inc.',
  BTC: 'Bitcoin',
  JFC: 'Jollibee Foods Corporation',
  LTC: 'Litecoin',
  PHR: 'PH Resorts Group Holdings, Inc.',
  SMI: 'SM Investments Corporation',
  UPM: 'United Paragon Mining Corporation',
  XRP: 'XRP'
}
const RECOMMENDATION_MAP = {
  '-1': 'Sell',
  '0': 'Hold',
  '1': 'Buy'
}

function listDirectories(dirPath) {
  return readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
}

function csvToRows(filePath) {
  const lines = readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  const [headerLine, ...dataLines] = lines
  const headers = headerLine.split(',')

  return dataLines.map((line) => {
    const values = line.split(',')
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']))
  })
}

function buildAssetRecords() {
  const assets = []
  const recommendations = []

  for (const category of listDirectories(stocksRoot)) {
    const categoryPath = path.join(stocksRoot, category)

    for (const assetCode of listDirectories(categoryPath)) {
      const predictionPath = path.join(categoryPath, assetCode, assetCode, PREDICTION_FILE)

      if (!statExists(predictionPath)) {
        continue
      }

      const categoryLabel = CATEGORY_LABELS[category] ?? category
      const fullName = ASSET_FULL_NAMES[assetCode] ?? assetCode
      const assetLabel = `${fullName} (${assetCode})`

      assets.push({
        code: assetCode,
        fullName,
        category,
        categoryLabel,
        label: assetLabel,
        predictionFile: predictionPath
      })

      for (const row of csvToRows(predictionPath)) {
        if (!RECOMMENDATION_MAP[row.y_pred]) {
          continue
        }

        recommendations.push({
          asset: assetCode,
          fullName,
          assetLabel,
          category,
          categoryLabel,
          date: row.Date,
          recommendation: RECOMMENDATION_MAP[row.y_pred],
          score: Number(row.p_buy)
        })
      }
    }
  }

  assets.sort((left, right) => left.code.localeCompare(right.code))
  recommendations.sort((left, right) => {
    if (left.asset === right.asset) {
      return left.date.localeCompare(right.date)
    }

    return left.asset.localeCompare(right.asset)
  })

  return { assets, recommendations }
}

function statExists(targetPath) {
  try {
    return statSync(targetPath).isFile()
  } catch {
    return false
  }
}

const { assets, recommendations } = buildAssetRecords()

mkdirSync(outputDir, { recursive: true })

const fileContents = `export const assetOptions = ${JSON.stringify(assets, null, 2)}\n\nexport const recommendationData = ${JSON.stringify(recommendations, null, 2)}\n`

writeFileSync(outputPath, fileContents, 'utf8')

console.log(`Generated ${recommendations.length} recommendations for ${assets.length} assets.`)
