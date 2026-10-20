import { initializeApp } from 'firebase/app'
import { getDatabase, ref, set } from 'firebase/database'

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  databaseURL: import.meta.env.VITE_FIREBASE_DATABASE_URL,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
}

const isConfigured = Boolean(
  firebaseConfig.apiKey && firebaseConfig.databaseURL,
)

let db = null

if (isConfigured) {
  const app = initializeApp(firebaseConfig)
  db = getDatabase(app)
}

/**
 * Save a completed session chat to Firebase Realtime Database.
 * Path: sessions/{userId}/{userId}-{savedAt}
 * The node key is "{userId}-{savedAtMs}" for easy sorting and lookup.
 */
export async function saveSessionToFirebase({ userId, sessionId, callId, taskId, savedAt, entries }) {
  if (!db) {
    console.warn('[Firebase] Not configured — skipping session upload.')
    return null
  }
  const uid = userId ?? 'anonymous'
  const ts = savedAt ?? Date.now()
  const nodeKey = `${uid}-${ts}`
  const payload = {
    userId: uid,
    sessionId: sessionId ?? null,
    callId: callId ?? null,
    taskId: taskId ?? null,
    savedAt: ts,
    entryCount: entries.length,
    entries,
  }
  const sessionRef = ref(db, `sessions/${uid}/${nodeKey}`)
  await set(sessionRef, payload)
  return nodeKey
}
