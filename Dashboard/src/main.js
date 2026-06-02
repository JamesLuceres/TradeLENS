import { createApp } from 'vue'
import { Quasar, Notify } from 'quasar'
import App from './App.vue'

import 'quasar/src/css/index.sass'
import '@quasar/extras/material-icons/material-icons.css'
import './css/app.css'

createApp(App)
  .use(Quasar, {
    plugins: {
      Notify
    }
  })
  .mount('#app')
