
<script setup>
import OwnerCameraPanel from './components/OwnerCameraPanel.vue'

import { computed, onMounted, reactive, ref } from 'vue'

const API_BASE = 'http://127.0.0.1:8000'

const tabs = [
  { key: 'dashboard', label: '系统总览' },
  { key: 'owner', label: '车主手势控车' },
  { key: 'plate', label: '车牌识别' },
  { key: 'traffic', label: '交警手势' },
  { key: 'stream', label: '视频流监控' },
  { key: 'records', label: '记录中心' },
]

const activeTab = ref('owner')

const backendStatus = ref({
  status: 'checking',
  message: '正在检测后端状态',
})

const dashboardSummary = ref(null)
const vehicleStateRaw = ref(null)
const records = ref([])
const alerts = ref([])
const logs = ref([])
const alertAnalysis = ref(null)
const monitorStatus = ref(null)

const loading = reactive({
  refresh: false,
  ownerImage: false,
  ownerVideo: false,
  plateImage: false,
  trafficImage: false,
  stream: false,
  monitor: false,
})

const errors = reactive({
  global: '',
  owner: '',
  plate: '',
  traffic: '',
  stream: '',
})

const ownerImageFile = ref(null)
const ownerVideoFile = ref(null)
const plateImageFile = ref(null)
const trafficImageFile = ref(null)

const ownerImageResult = ref(null)
const ownerVideoResult = ref(null)
const plateResult = ref(null)
const trafficResult = ref(null)
const streamResult = ref(null)

const ownerFrameInterval = ref(3)
const ownerStableThreshold = ref(3)

const streamForm = reactive({
  source_id: 'live12',
  task_type: 'plate',
  frame_count: 20,
  sample_interval: 5,
  use_mock_frame: false,
})

const monitorForm = reactive({
  task_type: 'plate',
  interval_seconds: 30,
  frame_count: 20,
  sample_interval: 5,
  use_mock_frame: false,
})

const ownerLatestResult = computed(() => ownerVideoResult.value || ownerImageResult.value)
const ownerResult = computed(() => ownerLatestResult.value?.result || {})
const ownerVehicleState = computed(() => {
  return ownerResult.value?.vehicle_state || vehicleStateRaw.value || {
    system_awake: false,
    current_function: 'home',
    volume: 50,
    temperature: 24,
    phone_status: '空闲',
    updated_at: '',
  }
})

const ownerLandmarks = computed(() => {
  const result = ownerResult.value
  if (Array.isArray(result.landmarks)) return result.landmarks
  if (Array.isArray(result.frame_results)) {
    const item = result.frame_results.find((frame) => Array.isArray(frame.landmarks) && frame.landmarks.length > 0)
    return item?.landmarks || []
  }
  return []
})

const ownerKeyLandmarks = computed(() => {
  const keyIds = new Set([0, 4, 8, 12, 16, 20])
  return ownerLandmarks.value.filter((item) => keyIds.has(item.index))
})

const supportedOwnerGestures = [
  { gesture: 'open_palm', name: '手掌张开', action: '启动 / 唤醒系统' },
  { gesture: 'fist', name: '握拳', action: '确认 / 执行当前功能' },
  { gesture: 'one', name: '单指', action: '音量增加' },
  { gesture: 'two', name: '双指', action: '音量降低' },
  { gesture: 'thumb_up', name: '拇指向上', action: '接听电话' },
  { gesture: 'thumb_down', name: '拇指向下', action: '挂断电话' },
  { gesture: 'swipe_left', name: '左滑', action: '切换上一个功能' },
  { gesture: 'swipe_right', name: '右滑', action: '切换下一个功能' },
  { gesture: 'wave', name: '挥手', action: '返回主页' },
  { gesture: 'circle', name: '单指画圈', action: '调节音量' },
]

function assetUrl(path) {
  if (!path) return ''
  if (path.startsWith('http')) return path
  return `${API_BASE}${path}`
}

function prettyJson(value) {
  if (!value) return ''
  return JSON.stringify(value, null, 2)
}

function percent(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '0%'
  if (num <= 1) return `${Math.round(num * 100)}%`
  return `${Math.round(num)}%`
}

function shortText(value, max = 80) {
  if (value === null || value === undefined) return '-'
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return text.length > max ? `${text.slice(0, max)}...` : text
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const text = await response.text()
  let data = {}

  try {
    data = text ? JSON.parse(text) : {}
  } catch {
    data = { raw: text }
  }

  if (!response.ok) {
    throw new Error(data?.detail || data?.message || text || `HTTP ${response.status}`)
  }

  return data
}

function pickFile(event, targetName) {
  const file = event.target.files?.[0] || null

  if (targetName === 'ownerImage') ownerImageFile.value = file
  if (targetName === 'ownerVideo') ownerVideoFile.value = file
  if (targetName === 'plateImage') plateImageFile.value = file
  if (targetName === 'trafficImage') trafficImageFile.value = file

  if (targetName === 'ownerImage' || targetName === 'ownerVideo') errors.owner = ''
  if (targetName === 'plateImage') errors.plate = ''
  if (targetName === 'trafficImage') errors.traffic = ''
}

async function refreshAll() {
  loading.refresh = true
  errors.global = ''

  try {
    const tasks = await Promise.allSettled([
      requestJson('/api/health'),
      requestJson('/api/dashboard/summary'),
      requestJson('/api/vehicle/state'),
      requestJson('/api/records?limit=10'),
      requestJson('/api/alerts?limit=10'),
      requestJson('/api/logs?limit=10'),
      requestJson('/api/alerts/analysis'),
      requestJson('/api/monitor/status'),
    ])

    const [health, summary, vehicle, recordData, alertData, logData, analysis, monitor] = tasks

    if (health.status === 'fulfilled') {
      backendStatus.value = health.value
    } else {
      backendStatus.value = {
        status: 'error',
        message: health.reason?.message || '后端连接失败',
      }
    }

    if (summary.status === 'fulfilled') dashboardSummary.value = summary.value
    if (vehicle.status === 'fulfilled') vehicleStateRaw.value = vehicle.value?.vehicle_state || vehicle.value
    if (recordData.status === 'fulfilled') records.value = recordData.value?.records || recordData.value?.items || []
    if (alertData.status === 'fulfilled') alerts.value = alertData.value?.alerts || alertData.value?.items || []
    if (logData.status === 'fulfilled') logs.value = logData.value?.logs || logData.value?.items || []
    if (analysis.status === 'fulfilled') alertAnalysis.value = analysis.value
    if (monitor.status === 'fulfilled') monitorStatus.value = monitor.value?.monitor || monitor.value
  } catch (error) {
    errors.global = error.message
  } finally {
    loading.refresh = false
  }
}

async function uploadOwnerImage() {
  if (!ownerImageFile.value) {
    errors.owner = '请先选择车主手势图片'
    return
  }

  loading.ownerImage = true
  errors.owner = ''

  try {
    const formData = new FormData()
    formData.append('file', ownerImageFile.value)

    ownerImageResult.value = await requestJson('/api/gesture/owner/image', {
      method: 'POST',
      body: formData,
    })

    ownerVideoResult.value = null
    await refreshAll()
  } catch (error) {
    errors.owner = error.message
  } finally {
    loading.ownerImage = false
  }
}

async function uploadOwnerVideo() {
  if (!ownerVideoFile.value) {
    errors.owner = '请先选择车主手势视频'
    return
  }

  loading.ownerVideo = true
  errors.owner = ''

  try {
    const formData = new FormData()
    formData.append('file', ownerVideoFile.value)

    const query = new URLSearchParams({
      frame_sample_interval: String(ownerFrameInterval.value),
      stable_threshold: String(ownerStableThreshold.value),
    })

    ownerVideoResult.value = await requestJson(`/api/gesture/owner/video?${query.toString()}`, {
      method: 'POST',
      body: formData,
    })

    ownerImageResult.value = null
    await refreshAll()
  } catch (error) {
    errors.owner = error.message
  } finally {
    loading.ownerVideo = false
  }
}

async function uploadPlateImage() {
  if (!plateImageFile.value) {
    errors.plate = '请先选择车牌图片'
    return
  }

  loading.plateImage = true
  errors.plate = ''

  try {
    const formData = new FormData()
    formData.append('file', plateImageFile.value)

    plateResult.value = await requestJson('/api/plate/image', {
      method: 'POST',
      body: formData,
    })

    await refreshAll()
  } catch (error) {
    errors.plate = error.message
  } finally {
    loading.plateImage = false
  }
}

async function uploadTrafficImage() {
  if (!trafficImageFile.value) {
    errors.traffic = '请先选择交警手势图片'
    return
  }

  loading.trafficImage = true
  errors.traffic = ''

  try {
    const formData = new FormData()
    formData.append('file', trafficImageFile.value)

    trafficResult.value = await requestJson('/api/gesture/traffic/image', {
      method: 'POST',
      body: formData,
    })

    await refreshAll()
  } catch (error) {
    errors.traffic = error.message
  } finally {
    loading.trafficImage = false
  }
}

async function recognizeStream() {
  loading.stream = true
  errors.stream = ''

  try {
    streamResult.value = await requestJson('/api/stream/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(streamForm),
    })

    await refreshAll()
  } catch (error) {
    errors.stream = error.message
  } finally {
    loading.stream = false
  }
}

async function startMonitor() {
  loading.monitor = true
  errors.stream = ''

  try {
    const body = {
      task_type: monitorForm.task_type,
      interval_seconds: Number(monitorForm.interval_seconds),
      frame_count: Number(monitorForm.frame_count),
      sample_interval: Number(monitorForm.sample_interval),
      use_mock_frame: Boolean(monitorForm.use_mock_frame),
      source_ids: ['all'],
    }

    const result = await requestJson('/api/monitor/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    monitorStatus.value = result.monitor || result
    await refreshAll()
  } catch (error) {
    errors.stream = error.message
  } finally {
    loading.monitor = false
  }
}

async function stopMonitor() {
  loading.monitor = true
  errors.stream = ''

  try {
    const result = await requestJson('/api/monitor/stop', {
      method: 'POST',
    })

    monitorStatus.value = result.monitor || result
    await refreshAll()
  } catch (error) {
    errors.stream = error.message
  } finally {
    loading.monitor = false
  }
}

onMounted(() => {
  refreshAll()
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">AI Intelligent Traffic Recognition System</p>
        <h1>智能交通识别与车载视觉感知系统</h1>
        <p class="subtitle">
          车牌识别、交警手势识别、车主手势控车、视频流监控与告警分析一体化演示平台
        </p>
      </div>

      <div class="status-card">
        <span
          class="status-dot"
          :class="{ ok: backendStatus.status === 'ok' || backendStatus.status === 'success' }"
        ></span>
        <div>
          <strong>{{ backendStatus.status || 'unknown' }}</strong>
          <p>{{ backendStatus.message || '后端状态未知' }}</p>
        </div>
        <button class="small-btn" :disabled="loading.refresh" @click="refreshAll">
          {{ loading.refresh ? '刷新中' : '刷新' }}
        </button>
      </div>
    </header>

    <nav class="nav-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </nav>

    <p v-if="errors.global" class="error-line">{{ errors.global }}</p>

    <main>
      <section v-if="activeTab === 'dashboard'" class="page-grid">
        <div class="panel span-12">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Dashboard</p>
              <h2>系统总览</h2>
            </div>
          </div>

          <div class="metric-grid">
            <div class="metric-card">
              <span>后端状态</span>
              <strong>{{ backendStatus.status }}</strong>
              <p>{{ backendStatus.message }}</p>
            </div>
            <div class="metric-card">
              <span>自动监控</span>
              <strong>{{ monitorStatus?.running ? '运行中' : '未运行' }}</strong>
              <p>{{ monitorStatus?.task_type || '暂无任务' }}</p>
            </div>
            <div class="metric-card">
              <span>车辆功能</span>
              <strong>{{ ownerVehicleState.current_function }}</strong>
              <p>音量 {{ ownerVehicleState.volume }} / 温度 {{ ownerVehicleState.temperature }}℃</p>
            </div>
            <div class="metric-card">
              <span>电话状态</span>
              <strong>{{ ownerVehicleState.phone_status }}</strong>
              <p>{{ ownerVehicleState.system_awake ? '系统已唤醒' : '系统未唤醒' }}</p>
            </div>
          </div>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <h2>告警智能分析</h2>
          </div>
          <pre class="json-box">{{ prettyJson(alertAnalysis || dashboardSummary) }}</pre>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <h2>最近识别记录</h2>
          </div>
          <div class="simple-list">
            <div v-for="item in records.slice(0, 6)" :key="item.id || item.record_id" class="list-item">
              <strong>#{{ item.id || item.record_id }} {{ item.task_type }}</strong>
              <span>{{ item.input_type }} · {{ shortText(item.created_at || item.created_time, 30) }}</span>
            </div>
          </div>
        </div>
      </section>

      <section v-if="activeTab === 'owner'" class="page-grid">
        <div class="panel span-12 hero-panel">
          <div>
            <p class="eyebrow">Owner Gesture Control</p>
            <h2>车主手势控车</h2>
      <OwnerCameraPanel />
            <p>
              支持车主手势图片与视频上传识别。后端基于 MediaPipe Hands 提取 21 个手部关键点，
              并通过连续帧确认与动态轨迹判断降低误触发。
            </p>
          </div>
          <div class="badge-row">
            <span>至少 6 种手势</span>
            <span>关键点骨架</span>
            <span>车辆状态联动</span>
            <span>误触发抑制</span>
          </div>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Image Input</p>
              <h2>上传车主手势图片</h2>
            </div>
          </div>

          <div class="upload-box">
            <input type="file" accept="image/*" @change="pickFile($event, 'ownerImage')" />
            <button :disabled="loading.ownerImage" @click="uploadOwnerImage">
              {{ loading.ownerImage ? '识别中...' : '开始图片识别' }}
            </button>
          </div>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Video Input</p>
              <h2>上传车主手势视频</h2>
            </div>
          </div>

          <div class="upload-box">
            <input type="file" accept="video/*" @change="pickFile($event, 'ownerVideo')" />

            <div class="form-row">
              <label>
                抽帧间隔
                <input v-model.number="ownerFrameInterval" type="number" min="1" max="30" />
              </label>
              <label>
                触发阈值
                <input v-model.number="ownerStableThreshold" type="number" min="1" max="20" />
              </label>
            </div>

            <button :disabled="loading.ownerVideo" @click="uploadOwnerVideo">
              {{ loading.ownerVideo ? '识别中...' : '开始视频识别' }}
            </button>
          </div>
        </div>

        <p v-if="errors.owner" class="error-line span-12">{{ errors.owner }}</p>

        <div class="panel span-5">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Recognition Result</p>
              <h2>识别结果</h2>
            </div>
          </div>

          <div v-if="ownerLatestResult" class="result-card">
            <div class="result-main">
              <span>{{ ownerResult.gesture_name || '暂无结果' }}</span>
              <strong>{{ ownerResult.gesture || '-' }}</strong>
            </div>

            <div class="kv-grid">
              <div>
                <span>置信度</span>
                <strong>{{ percent(ownerResult.confidence) }}</strong>
              </div>
              <div>
                <span>触发状态</span>
                <strong>{{ ownerResult.triggered === false ? '未触发' : '已触发 / 图片识别' }}</strong>
              </div>
              <div>
                <span>控制动作</span>
                <strong>{{ ownerResult.action || '-' }}</strong>
              </div>
              <div>
                <span>说明</span>
                <strong>{{ ownerResult.description || '-' }}</strong>
              </div>
            </div>

            <div v-if="ownerResult.trigger_policy" class="policy-box">
              <h3>误触发抑制</h3>
              <p><b>策略：</b>{{ ownerResult.trigger_policy }}</p>
              <p><b>阈值：</b>{{ ownerResult.stable_threshold }}</p>
              <p><b>原因：</b>{{ ownerResult.trigger_reason }}</p>
            </div>

            <div v-if="ownerResult.video_info" class="policy-box">
              <h3>视频抽帧信息</h3>
              <p>总帧数：{{ ownerResult.video_info.total_frames }}</p>
              <p>帧率：{{ ownerResult.video_info.fps }}</p>
              <p>读取帧数：{{ ownerResult.video_info.frames_read }}</p>
              <p>抽样帧数：{{ ownerResult.video_info.sampled_frames }}</p>
            </div>
          </div>

          <div v-else class="empty-state">
            上传车主手势图片或视频后，这里会显示识别结果。
          </div>
        </div>

        <div class="panel span-7">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Hand Landmark Skeleton</p>
              <h2>手部关键点骨架展示</h2>
            </div>
          </div>

          <div v-if="ownerLatestResult?.output_image_url" class="image-preview">
            <img :src="assetUrl(ownerLatestResult.output_image_url)" alt="车主手势识别标注图" />
          </div>
          <div v-else class="empty-preview">
            识别后将显示带有 MediaPipe 手部骨架的标注图。
          </div>

          <div v-if="ownerKeyLandmarks.length" class="landmark-strip">
            <div v-for="point in ownerKeyLandmarks" :key="point.index">
              <span>#{{ point.index }}</span>
              <strong>x {{ point.x }}</strong>
              <strong>y {{ point.y }}</strong>
            </div>
          </div>
        </div>

        <div class="panel span-5">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Vehicle Control Panel</p>
              <h2>模拟车辆控制面板</h2>
            </div>
          </div>

          <div class="vehicle-card">
            <div class="vehicle-row">
              <span>系统状态</span>
              <strong :class="ownerVehicleState.system_awake ? 'text-ok' : 'text-muted'">
                {{ ownerVehicleState.system_awake ? '已唤醒' : '未唤醒' }}
              </strong>
            </div>
            <div class="vehicle-row">
              <span>当前功能</span>
              <strong>{{ ownerVehicleState.current_function }}</strong>
            </div>
            <div class="vehicle-row">
              <span>电话状态</span>
              <strong>{{ ownerVehicleState.phone_status }}</strong>
            </div>
            <div class="vehicle-row">
              <span>空调温度</span>
              <strong>{{ ownerVehicleState.temperature }} ℃</strong>
            </div>

            <div class="slider-line">
              <div>
                <span>音量</span>
                <strong>{{ ownerVehicleState.volume }}</strong>
              </div>
              <div class="progress">
                <i :style="{ width: `${ownerVehicleState.volume || 0}%` }"></i>
              </div>
            </div>

            <p class="time-text">更新时间：{{ ownerVehicleState.updated_at || '-' }}</p>
          </div>
        </div>

        <div class="panel span-7">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Gesture Mapping</p>
              <h2>预定义手势与控制映射</h2>
            </div>
          </div>

          <div class="gesture-grid">
            <div v-for="item in supportedOwnerGestures" :key="item.gesture" class="gesture-item">
              <strong>{{ item.name }}</strong>
              <span>{{ item.gesture }}</span>
              <p>{{ item.action }}</p>
            </div>
          </div>
        </div>
      </section>

      <section v-if="activeTab === 'plate'" class="page-grid">
        <div class="panel span-6">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Plate Recognition</p>
              <h2>上传车牌图片识别</h2>
            </div>
          </div>

          <div class="upload-box">
            <input type="file" accept="image/*" @change="pickFile($event, 'plateImage')" />
            <button :disabled="loading.plateImage" @click="uploadPlateImage">
              {{ loading.plateImage ? '识别中...' : '开始识别' }}
            </button>
          </div>

          <p v-if="errors.plate" class="error-line">{{ errors.plate }}</p>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <h2>车牌识别结果</h2>
          </div>

          <div v-if="plateResult" class="result-card">
            <div class="image-preview">
              <img v-if="plateResult.output_image_url" :src="assetUrl(plateResult.output_image_url)" />
            </div>
            <pre class="json-box">{{ prettyJson(plateResult.result) }}</pre>
          </div>
          <div v-else class="empty-state">上传车牌图片后显示结果。</div>
        </div>
      </section>

      <section v-if="activeTab === 'traffic'" class="page-grid">
        <div class="panel span-6">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Traffic Police Gesture</p>
              <h2>上传交警手势图片</h2>
            </div>
          </div>

          <div class="upload-box">
            <input type="file" accept="image/*" @change="pickFile($event, 'trafficImage')" />
            <button :disabled="loading.trafficImage" @click="uploadTrafficImage">
              {{ loading.trafficImage ? '识别中...' : '开始识别' }}
            </button>
          </div>

          <p v-if="errors.traffic" class="error-line">{{ errors.traffic }}</p>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <h2>交警手势结果</h2>
          </div>

          <div v-if="trafficResult" class="result-card">
            <div class="image-preview">
              <img v-if="trafficResult.output_image_url" :src="assetUrl(trafficResult.output_image_url)" />
            </div>
            <pre class="json-box">{{ prettyJson(trafficResult.result) }}</pre>
          </div>
          <div v-else class="empty-state">上传交警手势图片后显示结果。</div>
        </div>
      </section>

      <section v-if="activeTab === 'stream'" class="page-grid">
        <div class="panel span-6">
          <div class="panel-header">
            <div>
              <p class="eyebrow">RTSP Stream</p>
              <h2>视频流识别</h2>
            </div>
          </div>

          <div class="form-grid">
            <label>
              视频源 ID
              <input v-model="streamForm.source_id" />
            </label>
            <label>
              任务类型
              <select v-model="streamForm.task_type">
                <option value="plate">车牌识别</option>
                <option value="owner_gesture">车主手势</option>
                <option value="traffic_gesture">交警手势</option>
              </select>
            </label>
            <label>
              读取帧数
              <input v-model.number="streamForm.frame_count" type="number" min="1" />
            </label>
            <label>
              抽样间隔
              <input v-model.number="streamForm.sample_interval" type="number" min="1" />
            </label>
            <label class="checkbox-line">
              <input v-model="streamForm.use_mock_frame" type="checkbox" />
              使用模拟帧
            </label>
          </div>

          <button :disabled="loading.stream" @click="recognizeStream">
            {{ loading.stream ? '识别中...' : '开始视频流识别' }}
          </button>

          <p v-if="errors.stream" class="error-line">{{ errors.stream }}</p>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Auto Monitor</p>
              <h2>全局自动监控</h2>
            </div>
          </div>

          <div class="form-grid">
            <label>
              任务类型
              <select v-model="monitorForm.task_type">
                <option value="plate">车牌识别</option>
                <option value="owner_gesture">车主手势</option>
                <option value="traffic_gesture">交警手势</option>
              </select>
            </label>
            <label>
              周期秒数
              <input v-model.number="monitorForm.interval_seconds" type="number" min="5" />
            </label>
            <label>
              每次读取帧数
              <input v-model.number="monitorForm.frame_count" type="number" min="1" />
            </label>
            <label>
              抽样间隔
              <input v-model.number="monitorForm.sample_interval" type="number" min="1" />
            </label>
            <label class="checkbox-line">
              <input v-model="monitorForm.use_mock_frame" type="checkbox" />
              使用模拟帧
            </label>
          </div>

          <div class="button-row">
            <button :disabled="loading.monitor" @click="startMonitor">启动监控</button>
            <button class="ghost-btn" :disabled="loading.monitor" @click="stopMonitor">停止监控</button>
          </div>

          <div class="policy-box">
            <p><b>当前状态：</b>{{ monitorStatus?.running ? '运行中' : '未运行' }}</p>
            <p><b>任务类型：</b>{{ monitorStatus?.task_type || '-' }}</p>
          </div>
        </div>

        <div class="panel span-12">
          <div class="panel-header">
            <h2>视频流识别结果</h2>
          </div>

          <div v-if="streamResult" class="result-card">
            <div class="image-preview">
              <img v-if="streamResult.output_image_url" :src="assetUrl(streamResult.output_image_url)" />
            </div>
            <pre class="json-box">{{ prettyJson(streamResult.result) }}</pre>
          </div>
          <div v-else class="empty-state">运行视频流识别后显示结果。</div>
        </div>
      </section>

      <section v-if="activeTab === 'records'" class="page-grid">
        <div class="panel span-12">
          <div class="panel-header">
            <div>
              <p class="eyebrow">Records</p>
              <h2>识别记录</h2>
            </div>
            <button class="small-btn" @click="refreshAll">刷新</button>
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>任务</th>
                  <th>输入</th>
                  <th>文件</th>
                  <th>结果摘要</th>
                  <th>时间</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in records" :key="item.id || item.record_id">
                  <td>{{ item.id || item.record_id }}</td>
                  <td>{{ item.task_type }}</td>
                  <td>{{ item.input_type }}</td>
                  <td>{{ shortText(item.original_filename || item.saved_filename, 30) }}</td>
                  <td>{{ shortText(item.result, 100) }}</td>
                  <td>{{ item.created_at || item.created_time || '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <h2>告警记录</h2>
          </div>

          <div class="simple-list">
            <div v-for="item in alerts" :key="item.id || item.alert_id" class="list-item">
              <strong>#{{ item.id || item.alert_id }} {{ item.alert_type || item.type || 'alert' }}</strong>
              <span>{{ shortText(item.description || item.detail || item.message, 100) }}</span>
            </div>
          </div>
        </div>

        <div class="panel span-6">
          <div class="panel-header">
            <h2>操作日志</h2>
          </div>

          <div class="simple-list">
            <div v-for="item in logs" :key="item.id || item.log_id" class="list-item">
              <strong>#{{ item.id || item.log_id }} {{ item.action }}</strong>
              <span>{{ shortText(item.detail || item.created_at || item.created_time, 100) }}</span>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>
