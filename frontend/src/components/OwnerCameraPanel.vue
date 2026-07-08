<template>
  <section class="owner-camera-panel">
    <div class="owner-camera-header">
      <div>
        <h3>电脑摄像头实时识别</h3>
        <p>
          低分辨率快速抽帧上传至后端 MediaPipe Hands 接口，前端本地绘制手部关键点骨架，优先保证实时性。
        </p>
      </div>
      <div class="owner-camera-tags">
        <span :class="['camera-tag', cameraRunning ? 'ok' : 'warn']">
          {{ cameraRunning ? '摄像头已启动' : '摄像头未启动' }}
        </span>
        <span :class="['camera-tag', loopRunning ? 'ok' : 'warn']">
          {{ loopRunning ? '实时识别中' : '实时识别未开始' }}
        </span>
        <span class="camera-tag ok" v-if="fastMode">快速模式</span>
      </div>
    </div>

    <div class="owner-camera-grid">
      <div class="owner-camera-card">
        <div class="owner-video-stage">
          <video ref="videoRef" autoplay muted playsinline></video>
          <canvas ref="overlayCanvasRef" class="owner-overlay-canvas"></canvas>
          <canvas ref="captureCanvasRef" class="owner-hidden-canvas"></canvas>
        </div>

        <div class="owner-camera-controls">
          <button type="button" @click="startCamera" :disabled="cameraRunning">
            打开摄像头
          </button>
          <button type="button" class="secondary" @click="stopCamera" :disabled="!cameraRunning">
            关闭摄像头
          </button>
          <button type="button" class="secondary" @click="captureAndRecognize" :disabled="!cameraRunning || requesting">
            单次识别
          </button>
          <button type="button" @click="startLoop" :disabled="!cameraRunning || loopRunning">
            开始实时识别
          </button>
          <button type="button" class="danger" @click="stopLoop" :disabled="!loopRunning">
            停止实时识别
          </button>

          <label class="camera-inline-field">
            间隔 ms
            <input v-model.number="intervalMs" type="number" min="250" step="50" />
          </label>

          <label class="camera-inline-field">
            帧宽
            <select v-model.number="frameWidth">
              <option :value="240">240</option>
              <option :value="320">320</option>
              <option :value="400">400</option>
            </select>
          </label>
        </div>

        <p v-if="errorMessage" class="owner-camera-error">{{ errorMessage }}</p>
      </div>

      <div class="owner-camera-card">
        <div class="owner-result-main">
          <span>当前手势</span>
          <strong>{{ result.gesture_name || '暂无结果' }}</strong>
        </div>

        <div class="owner-camera-metrics">
          <div>
            <span>手势代码</span>
            <strong>{{ result.gesture || '-' }}</strong>
          </div>
          <div>
            <span>置信度</span>
            <strong>{{ confidenceText }}</strong>
          </div>
          <div>
            <span>车辆动作</span>
            <strong>{{ result.description || result.action || '-' }}</strong>
          </div>
          <div>
            <span>接口延迟</span>
            <strong>{{ latencyText }}</strong>
          </div>
          <div>
            <span>系统唤醒</span>
            <strong>{{ vehicleState.system_awake ? '已唤醒' : '未唤醒' }}</strong>
          </div>
          <div>
            <span>当前功能</span>
            <strong>{{ vehicleState.current_function || '-' }}</strong>
          </div>
          <div>
            <span>音量</span>
            <strong>{{ vehicleState.volume ?? '-' }}</strong>
          </div>
          <div>
            <span>电话状态</span>
            <strong>{{ vehicleState.phone_status || '-' }}</strong>
          </div>
        </div>

        <div class="owner-camera-log">
          <div v-for="item in logs" :key="item.id">
            {{ item.text }}
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, reactive, ref } from 'vue'

const API_BASE = 'http://127.0.0.1:8000'

const videoRef = ref(null)
const overlayCanvasRef = ref(null)
const captureCanvasRef = ref(null)

const streamRef = ref(null)
const cameraRunning = ref(false)
const loopRunning = ref(false)
const requesting = ref(false)
const fastMode = ref(true)

const intervalMs = ref(350)
const frameWidth = ref(320)

const loopTimer = ref(null)
const frameCount = ref(0)
const successCount = ref(0)
const totalLatency = ref(0)

const errorMessage = ref('')
const logs = ref([])

const result = reactive({
  gesture: '',
  gesture_name: '',
  confidence: null,
  action: '',
  description: '',
  latency_ms: null,
  landmarks: [],
})

const vehicleState = reactive({
  system_awake: false,
  current_function: 'home',
  volume: 50,
  temperature: 24,
  phone_status: '空闲',
})

const confidenceText = computed(() => {
  if (result.confidence === null || result.confidence === undefined) return '-'
  return Number(result.confidence).toFixed(2)
})

const latencyText = computed(() => {
  if (result.latency_ms === null || result.latency_ms === undefined) return '-'
  return `${Math.round(result.latency_ms)} ms`
})

function addLog(text) {
  const now = new Date().toLocaleTimeString()
  logs.value.unshift({
    id: `${Date.now()}_${Math.random()}`,
    text: `[${now}] ${text}`,
  })

  if (logs.value.length > 8) {
    logs.value = logs.value.slice(0, 8)
  }
}

async function startCamera() {
  errorMessage.value = ''

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 320 },
        height: { ideal: 240 },
        facingMode: 'user',
      },
      audio: false,
    })

    streamRef.value = stream
    videoRef.value.srcObject = stream

    cameraRunning.value = true

    await nextTick()
    resizeOverlayCanvas()

    addLog('摄像头已启动')
  } catch (error) {
    errorMessage.value = `摄像头启动失败：${error.message}`
    addLog(`摄像头启动失败：${error.message}`)
  }
}

function stopCamera() {
  stopLoop()

  if (streamRef.value) {
    streamRef.value.getTracks().forEach((track) => track.stop())
    streamRef.value = null
  }

  if (videoRef.value) {
    videoRef.value.srcObject = null
  }

  cameraRunning.value = false
  clearOverlay()
  addLog('摄像头已关闭')
}

function startLoop() {
  if (!cameraRunning.value) {
    errorMessage.value = '请先打开摄像头。'
    return
  }

  stopLoop()

  const delay = Math.max(250, Number(intervalMs.value || 350))
  intervalMs.value = delay

  loopRunning.value = true
  captureAndRecognize()
  loopTimer.value = window.setInterval(captureAndRecognize, delay)

  addLog(`实时识别已开始，间隔 ${delay}ms`)
}

function stopLoop() {
  if (loopTimer.value) {
    window.clearInterval(loopTimer.value)
    loopTimer.value = null
  }

  loopRunning.value = false
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.5)
  })
}

async function captureAndRecognize() {
  if (!cameraRunning.value || requesting.value) return

  const video = videoRef.value
  const captureCanvas = captureCanvasRef.value

  if (!video || !captureCanvas || !video.videoWidth || !video.videoHeight) return

  requesting.value = true
  errorMessage.value = ''

  try {
    const maxWidth = Math.max(160, Number(frameWidth.value || 320))
    const scale = Math.min(1, maxWidth / video.videoWidth)

    captureCanvas.width = Math.round(video.videoWidth * scale)
    captureCanvas.height = Math.round(video.videoHeight * scale)

    const ctx = captureCanvas.getContext('2d')
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height)

    const blob = await canvasToBlob(captureCanvas)
    if (!blob) throw new Error('摄像头画面转图片失败')

    const formData = new FormData()
    formData.append('file', blob, `owner_camera_${Date.now()}.jpg`)

    const clientStart = performance.now()

    const response = await fetch(`${API_BASE}/api/gesture/owner/camera-fast-frame`, {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()
    const clientLatency = performance.now() - clientStart

    if (!response.ok || data.status !== 'success') {
      throw new Error(JSON.stringify(data))
    }

    updateResult(data, clientLatency)
  } catch (error) {
    errorMessage.value = `识别失败：${error.message}`
    addLog(`识别失败：${error.message}`)
  } finally {
    requesting.value = false
  }
}

function updateResult(data, clientLatency) {
  const payload = data.result || {}
  const state = payload.vehicle_state || {}

  frameCount.value += 1
  successCount.value += 1

  const latency = data.latency_ms ?? clientLatency
  totalLatency.value += latency

  result.gesture = payload.gesture || ''
  result.gesture_name = payload.gesture_name || ''
  result.confidence = payload.confidence
  result.action = payload.action || ''
  result.description = payload.description || ''
  result.latency_ms = latency
  result.landmarks = payload.landmarks || []

  Object.assign(vehicleState, state)

  drawLandmarks(result.landmarks)

  const avg = successCount.value > 0 ? Math.round(totalLatency.value / successCount.value) : 0
  addLog(
    `第 ${frameCount.value} 帧：${result.gesture_name || '-'} / ${result.gesture || '-'}，延迟 ${Math.round(latency)}ms，平均 ${avg}ms`
  )
}

function resizeOverlayCanvas() {
  const video = videoRef.value
  const canvas = overlayCanvasRef.value

  if (!video || !canvas) return

  const rect = video.getBoundingClientRect()
  canvas.width = Math.round(rect.width)
  canvas.height = Math.round(rect.height)
}

function clearOverlay() {
  const canvas = overlayCanvasRef.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

function drawLandmarks(landmarks) {
  resizeOverlayCanvas()

  const canvas = overlayCanvasRef.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (!landmarks || landmarks.length === 0) return

  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ]

  const getPoint = (idx) => {
    const item = landmarks.find((lm) => lm.index === idx)
    if (!item) return null

    return {
      x: item.x * canvas.width,
      y: item.y * canvas.height,
    }
  }

  ctx.lineWidth = 3
  ctx.strokeStyle = '#38bdf8'
  ctx.fillStyle = '#f97316'

  for (const [a, b] of connections) {
    const p1 = getPoint(a)
    const p2 = getPoint(b)
    if (!p1 || !p2) continue

    ctx.beginPath()
    ctx.moveTo(p1.x, p1.y)
    ctx.lineTo(p2.x, p2.y)
    ctx.stroke()
  }

  for (const lm of landmarks) {
    ctx.beginPath()
    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, Math.PI * 2)
    ctx.fill()
  }
}

onBeforeUnmount(() => {
  stopCamera()
})

window.addEventListener('resize', resizeOverlayCanvas)
</script>

<style scoped>
.owner-camera-panel {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 18px;
  background: rgba(15, 23, 42, 0.05);
}

.owner-camera-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 16px;
}

.owner-camera-header h3 {
  margin: 0 0 6px;
  font-size: 20px;
}

.owner-camera-header p {
  margin: 0;
  color: #64748b;
  line-height: 1.6;
}

.owner-camera-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.camera-tag {
  display: inline-flex;
  align-items: center;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 12px;
  background: #e2e8f0;
  color: #334155;
}

.camera-tag.ok {
  background: #dcfce7;
  color: #166534;
}

.camera-tag.warn {
  background: #fef3c7;
  color: #92400e;
}

.owner-camera-grid {
  display: grid;
  grid-template-columns: minmax(320px, 1.2fr) minmax(300px, 1fr);
  gap: 16px;
}

.owner-camera-card {
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 16px;
  padding: 14px;
  background: rgba(255, 255, 255, 0.74);
}

.owner-video-stage {
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: 14px;
  background: #020617;
  border: 1px solid #cbd5e1;
}

.owner-video-stage video {
  display: block;
  width: 100%;
  min-height: 260px;
  object-fit: cover;
}

.owner-overlay-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.owner-hidden-canvas {
  display: none;
}

.owner-camera-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  align-items: center;
}

.owner-camera-controls button {
  border: none;
  border-radius: 10px;
  padding: 9px 12px;
  background: #2563eb;
  color: #ffffff;
  font-weight: 700;
  cursor: pointer;
}

.owner-camera-controls button.secondary {
  background: #475569;
}

.owner-camera-controls button.danger {
  background: #dc2626;
}

.owner-camera-controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.camera-inline-field {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #475569;
  font-size: 13px;
}

.camera-inline-field input,
.camera-inline-field select {
  width: 82px;
  border: 1px solid #cbd5e1;
  border-radius: 9px;
  padding: 8px;
  background: #ffffff;
}

.owner-camera-error {
  margin: 10px 0 0;
  color: #dc2626;
  white-space: pre-wrap;
}

.owner-result-main {
  display: grid;
  gap: 6px;
  margin-bottom: 12px;
}

.owner-result-main span {
  color: #64748b;
  font-size: 13px;
}

.owner-result-main strong {
  font-size: 26px;
}

.owner-camera-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.owner-camera-metrics div {
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 12px;
  padding: 10px;
  background: #f8fafc;
}

.owner-camera-metrics span {
  display: block;
  color: #64748b;
  font-size: 12px;
  margin-bottom: 5px;
}

.owner-camera-metrics strong {
  display: block;
  color: #0f172a;
  font-size: 15px;
  word-break: break-all;
}

.owner-camera-log {
  margin-top: 12px;
  height: 150px;
  overflow: auto;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.28);
  background: #0f172a;
  color: #dbeafe;
  padding: 10px;
  font-family: Consolas, monospace;
  font-size: 12px;
  line-height: 1.6;
}

@media (max-width: 980px) {
  .owner-camera-header {
    flex-direction: column;
  }

  .owner-camera-tags {
    justify-content: flex-start;
  }

  .owner-camera-grid {
    grid-template-columns: 1fr;
  }
}
</style>
