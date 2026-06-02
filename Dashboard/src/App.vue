<template>
  <q-layout view="lHh Lpr lFf" class="dashboard-shell">
    <q-page-container>
      <q-page class="flex flex-center q-pa-md q-pa-lg-xl">
        <div class="dashboard-wrapper full-width">
          <div class="text-center q-mb-xl">
            <div class="text-overline text-weight-bold text-primary">Investment Signal Lookup</div>
            <h1 class="dashboard-title q-my-sm">TradeLENS: Buy, Hold, or Sell?</h1>
            <p class="dashboard-subtitle q-mx-auto">
              Select an asset from your thesis dataset and choose a valid test date to check the model recommendation.
            </p>
          </div>

          <div class="row q-col-gutter-lg items-stretch">
            <div class="col-12 col-md-5">
              <q-card flat bordered class="panel-card full-height">
                <q-card-section>
                  <div class="text-h6 text-weight-bold q-mb-sm">Search Inputs</div>
                  <div class="text-body2 text-grey-7 q-mb-lg">
                    Use the controls below to query the available recommendation records.
                  </div>

                  <q-select
                    v-model="selectedAsset"
                    :options="assetOptions"
                    label="Select stock or asset"
                    outlined
                    emit-value
                    map-options
                    options-dense
                    class="q-mb-md"
                  >
                    <template #prepend>
                      <q-icon name="candlestick_chart" />
                    </template>
                  </q-select>

                  <div v-if="selectedAssetMeta" class="text-caption text-grey-7 q-mb-md">
                    {{ selectedAssetMeta.categoryLabel }} dataset
                    <span class="text-weight-medium">•</span>
                    {{ availableDates.length }} available dates
                    <span class="text-weight-medium">•</span>
                    {{ availableDates[0] }} to {{ availableDates[availableDates.length - 1] }}
                  </div>

                  <q-input
                    v-model="selectedDate"
                    label="Select date"
                    outlined
                    readonly
                    :disable="!selectedAsset"
                    :hint="selectedAsset ? 'Pick a date from the calendar. Only available dataset dates can be selected.' : 'Select an asset first.'"
                    class="q-mb-lg"
                  >
                    <template #prepend>
                      <q-icon name="event" />
                    </template>
                    <template #append>
                      <q-icon name="calendar_month" class="cursor-pointer">
                        <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                          <q-date
                            v-model="selectedDate"
                            mask="YYYY-MM-DD"
                            color="primary"
                            bordered
                            emit-immediately
                            first-day-of-week="1"
                            :options="isDateAvailable"
                            :events="hasCalendarEvent"
                            event-color="primary"
                            :default-year-month="calendarDefaultYearMonth"
                            :navigation-min-year-month="calendarMinYearMonth"
                            :navigation-max-year-month="calendarMaxYearMonth"
                          >
                            <div class="row items-center justify-end q-gutter-sm q-pa-sm">
                              <q-btn v-close-popup label="Close" flat color="primary" />
                            </div>
                          </q-date>
                        </q-popup-proxy>
                      </q-icon>
                    </template>
                  </q-input>

                  <q-btn
                    label="Check Recommendation"
                    color="primary"
                    unelevated
                    no-caps
                    class="full-width q-py-sm"
                    @click="checkRecommendation"
                  />
                </q-card-section>
              </q-card>
            </div>

            <div class="col-12 col-md-7">
              <q-card flat bordered class="panel-card result-card full-height">
                <q-card-section class="full-height column justify-center">
                  <template v-if="resultState === 'idle'">
                    <div class="text-center">
                      <q-icon name="insights" size="72px" color="primary" />
                      <div class="text-h6 text-weight-medium q-mt-md">Ready to search</div>
                      <p class="text-body1 text-grey-7 q-mb-none">
                        Pick an asset and date, then click the button to view the recommendation.
                      </p>
                    </div>
                  </template>

                  <template v-else-if="resultState === 'missing-input'">
                    <div class="text-center">
                      <q-banner inline-actions rounded class="bg-grey-2 text-grey-9">
                        Please select both an asset and a date before checking the recommendation.
                      </q-banner>
                    </div>
                  </template>

                  <template v-else-if="resultState === 'not-found'">
                    <div class="text-center">
                      <q-icon name="event_busy" size="72px" color="grey-6" />
                      <div class="text-h6 q-mt-md">No data available for this date</div>
                      <p class="text-body1 text-grey-7 q-mb-none">
                        There is no recommendation stored for {{ selectedAssetMeta?.label ?? selectedAsset }} on {{ selectedDate }}.
                      </p>
                    </div>
                  </template>

                  <template v-else>
                    <q-card flat :class="['recommendation-card', recommendationMeta.bgClass]">
                      <q-card-section class="text-center q-py-xl">
                        <div class="text-overline text-weight-bold">{{ selectedAssetMeta?.label ?? selectedAsset }}</div>
                        <div class="text-subtitle1 q-mb-md">{{ selectedDate }}</div>
                        <div class="text-h3 text-weight-bold q-mb-sm">
                          {{ recommendationMeta.label }}
                        </div>
                        <div class="text-body1">
                          Model recommendation for the selected asset and date.
                        </div>
                      </q-card-section>
                    </q-card>
                  </template>
                </q-card-section>
              </q-card>
            </div>
          </div>
        </div>
      </q-page>
    </q-page-container>
  </q-layout>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import {
  assetOptions as generatedAssetOptions,
  recommendationData
} from './data/generatedRecommendations'

const selectedAsset = ref(null)
const selectedDate = ref('')
const resultState = ref('idle')
const recommendation = ref('')
const assetOptions = generatedAssetOptions.map((item) => ({
  label: item.label,
  value: item.code
}))

const recommendationMap = {
  Buy: {
    label: 'Buy',
    bgClass: 'bg-positive text-white'
  },
  Hold: {
    label: 'Hold',
    bgClass: 'bg-warning text-dark'
  },
  Sell: {
    label: 'Sell',
    bgClass: 'bg-negative text-white'
  }
}

const recommendationMeta = computed(() => recommendationMap[recommendation.value] ?? recommendationMap.Hold)
const selectedAssetMeta = computed(() =>
  generatedAssetOptions.find((item) => item.code === selectedAsset.value) ?? null
)
const availableDates = computed(() =>
  recommendationData
    .filter((item) => item.asset === selectedAsset.value)
    .map((item) => item.date)
)
const availableDateSet = computed(() => new Set(availableDates.value))
const calendarDefaultYearMonth = computed(() => {
  const baseDate = selectedDate.value || availableDates.value[0]
  return baseDate ? baseDate.slice(0, 7).replace('-', '/') : void 0
})
const calendarMinYearMonth = computed(() =>
  availableDates.value[0] ? availableDates.value[0].slice(0, 7).replace('-', '/') : void 0
)
const calendarMaxYearMonth = computed(() => {
  const lastDate = availableDates.value[availableDates.value.length - 1]
  return lastDate ? lastDate.slice(0, 7).replace('-', '/') : void 0
})

watch(selectedAsset, () => {
  selectedDate.value = ''
  recommendation.value = ''
  resultState.value = 'idle'
})

function checkRecommendation() {
  if (!selectedAsset.value || !selectedDate.value) {
    resultState.value = 'missing-input'
    recommendation.value = ''
    return
  }

  const match = recommendationData.find(
    (item) => item.asset === selectedAsset.value && item.date === selectedDate.value
  )

  if (!match) {
    resultState.value = 'not-found'
    recommendation.value = ''
    return
  }

  recommendation.value = match.recommendation
  resultState.value = 'found'
}

function normalizeCalendarDate(date) {
  return typeof date === 'string' ? date.replaceAll('/', '-') : ''
}

function isDateAvailable(date) {
  return availableDateSet.value.has(normalizeCalendarDate(date))
}

function hasCalendarEvent(date) {
  return availableDateSet.value.has(normalizeCalendarDate(date))
}
</script>
