AI智能交通识别系统 - 环境配置与软件说明
版本：阶段性开发说明
适用目录：D:\code\xiaoxueqi

============================================================
一、项目简介
============================================================

本项目是一个基于视觉识别的智能交通识别系统，当前采用前后端分离架构：

1. 后端：FastAPI + SQLite + OpenCV + HyperLPR3 + MediaPipe
2. 前端：Vue 3 + Vite
3. 数据库：SQLite，本地文件位于 backend/data/app.db
4. 主要功能：
   - 图片车牌识别
   - RTSP 视频流连续帧识别
   - 全局视频源自动监控
   - 车主手势识别与车辆控制模拟
   - 交警手势识别
   - 识别记录、告警记录、操作日志、告警智能分析
   - 车牌 YOLO 数据集标注、转换与训练准备
   - 车主手势、交警手势训练脚本骨架

当前项目已能完成基本系统演示。车牌正式训练建议等沙盘车开始移动后，重新采集动态场景数据再进行。

============================================================
二、推荐开发环境
============================================================

操作系统：
- Windows 10 或 Windows 11

推荐开发工具：
- Visual Studio Code
- PowerShell
- Git
- GitHub Desktop 可选，但不强制

浏览器：
- Chrome 或 Edge

后端运行环境：
- Python 3.11 推荐
- Python 虚拟环境：backend/.venv

前端运行环境：
- Node.js LTS 版本
- npm

数据库：
- 当前使用 SQLite，无需单独安装数据库服务
- MySQL 可作为后续扩展，目前不是必须

标注工具：
- LabelMe，用于车牌目标框标注
- 不再推荐使用 LabelImg，因为在当前 PyQt 环境下容易出现兼容问题

可选训练工具：
- Ultralytics YOLO，用于后续沙盘车牌检测模型训练
- scikit-learn，用于手势关键点分类器训练

============================================================
三、项目目录结构说明
============================================================

项目根目录：
D:\code\xiaoxueqi

主要目录：

algorithm/
  早期算法测试脚本，例如 RTSP 测试脚本。

backend/
  后端主目录。

backend/main.py
  FastAPI 后端入口文件。

backend/algorithm/
  算法模块目录，包括：
  - plate_recognizer.py：车牌识别模块
  - owner_gesture_recognizer.py：车主手势识别模块
  - traffic_gesture_recognizer.py：交警手势识别模块

backend/data/
  SQLite 数据库目录。
  注意：数据库文件不建议提交到 GitHub。

backend/uploads/
  接口上传图片保存目录。
  注意：运行产生的数据，不建议提交到 GitHub。

backend/outputs/
  标注识别结果图片输出目录。
  注意：运行产生的数据，不建议提交到 GitHub。

backend/datasets/
  数据集目录。
  包括沙盘车牌采集图片、LabelMe 标注文件、YOLO 数据集等。
  注意：正式数据集一般不提交 GitHub。

backend/models/
  训练模型保存目录。
  注意：模型文件一般不提交 GitHub。

backend/training/
  训练与数据处理脚本目录。

frontend/
  Vue 3 前端项目目录。

frontend/src/App.vue
  当前前端主页面。

demo/
  演示用图片目录，例如：
  - test.png：车牌图片测试
  - hand.jpg：车主手势测试
  - traffic.png：交警手势测试

docs/
  项目说明、数据集说明、标注方案等文档目录。

============================================================
四、后端环境配置
============================================================

1. 进入后端目录

cd D:\code\xiaoxueqi\backend

2. 创建虚拟环境，若已创建可跳过

python -m venv .venv

3. 激活虚拟环境

.\.venv\Scripts\activate

激活成功后，PowerShell 前面会出现：
(.venv)

4. 安装后端依赖

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

5. 启动后端

python -m uvicorn main:app --reload

后端默认地址：
http://127.0.0.1:8000

6. 健康检查

curl.exe "http://127.0.0.1:8000/api/health"

正常返回示例：
{"status":"ok","message":"backend is running"}

============================================================
五、前端环境配置
============================================================

1. 进入前端目录

cd D:\code\xiaoxueqi\frontend

2. 安装前端依赖

npm install

3. 启动前端

npm run dev

4. 浏览器访问

http://localhost:5173/

注意：
如果 http://127.0.0.1:5173/ 打不开，可优先使用 http://localhost:5173/。

============================================================
六、当前主要接口
============================================================

后端主要接口包括：

GET  /api/health
GET  /api/self-check
GET  /api/dashboard/summary
GET  /api/rtsp/sources
POST /api/rtsp/recognize
POST /api/stream/recognize
POST /api/plate/image
POST /api/gesture/owner/image
POST /api/gesture/owner/simulate
GET  /api/vehicle/state
POST /api/gesture/traffic/image
POST /api/gesture/traffic/simulate
GET  /api/records
GET  /api/alerts
POST /api/alerts/test
POST /api/alerts/{alert_id}/resolve
GET  /api/alerts/analysis
GET  /api/logs
POST /api/monitor/start
POST /api/monitor/stop
GET  /api/monitor/status

============================================================
七、常用测试命令
============================================================

1. 健康检查

curl.exe "http://127.0.0.1:8000/api/health"

2. 图片车牌识别

curl.exe -X POST "http://127.0.0.1:8000/api/plate/image" -F "file=@D:\code\xiaoxueqi\demo\test.png"

3. 车主手势识别

curl.exe -X POST "http://127.0.0.1:8000/api/gesture/owner/image" -F "file=@D:\code\xiaoxueqi\demo\hand.jpg"

4. 交警手势识别

curl.exe -X POST "http://127.0.0.1:8000/api/gesture/traffic/image" -F "file=@D:\code\xiaoxueqi\demo\traffic.png"

5. 模拟视频流识别

$body = @{
  source_id = "live12"
  task_type = "plate"
  frame_count = 20
  sample_interval = 5
  use_mock_frame = $true
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/stream/recognize" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

6. 真实 RTSP 视频流识别

$body = @{
  source_id = "live12"
  task_type = "plate"
  frame_count = 20
  sample_interval = 5
  use_mock_frame = $false
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/stream/recognize" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

7. 启动自动监控

$body = @{
  task_type = "plate"
  interval_seconds = 30
  frame_count = 20
  sample_interval = 5
  use_mock_frame = $false
  source_ids = @("all")
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/monitor/start" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

8. 查看自动监控状态

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/monitor/status" `
  -Method Get

9. 停止自动监控

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/monitor/stop" `
  -Method Post

============================================================
八、RTSP 沙盘视频流说明
============================================================

当前测试过的视频源示例：

rtsp://10.126.59.120:8554/live/live12

测试网络连通性：

ping 10.126.59.120
Test-NetConnection 10.126.59.120 -Port 8554

如果 TcpTestSucceeded 为 True，说明 RTSP 端口可连接。

说明：
当前 HyperLPR3 对普通车牌图片识别效果较好，但对沙盘车牌适配不足。真实 RTSP 能成功读取帧，但可能识别不到车牌。这不是接口失败，而是模型域适配问题。后续应等沙盘车辆移动后重新采集动态帧，并训练沙盘车牌检测模型。

============================================================
九、数据库与测试记录清理
============================================================

1. 清空数据库测试记录

cd D:\code\xiaoxueqi\backend
.\.venv\Scripts\activate
python clear_test_data.py

该脚本用于清空：
- 识别记录
- 告警记录
- 操作日志

并备份当前数据库。

2. 清空上传图片和输出图片

Remove-Item .\uploads\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\outputs\* -Recurse -Force -ErrorAction SilentlyContinue

============================================================
十、LabelMe 标注环境
============================================================

1. 安装 LabelMe

cd D:\code\xiaoxueqi\backend
.\.venv\Scripts\activate
python -m pip install labelme -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 启动 LabelMe

labelme

如果失败，使用：

python -m labelme

3. 车牌标注原则

- 使用 Rectangle 矩形框
- 类别统一写 plate
- 只框车牌本身，不框整车
- 一张图中有几个清晰车牌，就标几个 plate 框
- 太糊、太小、边界看不清的车牌先跳过

4. 推荐目录

原始图片：
backend/datasets/plate_sandbox/images

标注文件：
backend/datasets/plate_sandbox/annotations

============================================================
十一、沙盘车牌训练准备
============================================================

当前阶段已经具备：

1. RTSP 沙盘帧采集脚本
2. LabelMe 标注流程
3. LabelMe JSON 转 YOLO 数据集脚本
4. YOLO 标注可视化检查脚本
5. YOLO 训练命令预案

采集沙盘帧命令示例：

cd D:\code\xiaoxueqi

python backend\training\plate\collect_sandbox_frames.py `
  --rtsp-url "rtsp://10.126.59.120:8554/live/live12" `
  --source-id "live12" `
  --frame-count 1000 `
  --sample-interval 5 `
  --warmup-frames 10

LabelMe JSON 转 YOLO：

python backend\training\plate\prepare_yolo_dataset.py `
  --images-dir backend\datasets\plate_sandbox\images `
  --annotations-dir backend\datasets\plate_sandbox\annotations `
  --annotation-format json `
  --output-dir backend\datasets\plate_sandbox_yolo `
  --class-name plate `
  --train-ratio 0.8 `
  --val-ratio 0.2 `
  --test-ratio 0.0 `
  --clean

可视化检查：

python backend\training\plate\visualize_yolo_labels.py `
  --dataset-dir backend\datasets\plate_sandbox_yolo `
  --split train `
  --max-images 30

训练 YOLOv8 示例：

python -m pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

yolo detect train `
  data=backend/datasets/plate_sandbox_yolo/data.yaml `
  model=yolov8n.pt `
  imgsz=640 `
  epochs=50 `
  batch=4 `
  name=sandbox_plate_yolov8n

说明：
如果沙盘车辆当前静止，采集到的图片高度重复，不建议继续大量标注或训练。应等车辆开始移动后，重新采集动态场景数据。

============================================================
十二、车主手势数据集说明
============================================================

老师提供的两个车主手势相关数据集：

1. HaGRID
地址：https://github.com/hukenovs/hagrid
用途：静态手势图片识别，可用于车主静态手势分类训练。

2. NVGesture
地址：https://opendatalab.com/OpenDataLab/NVGesture
用途：动态车载手势识别，可用于后续动态手势增强。

当前建议：
- 优先完善基于 MediaPipe Hands 的车主手势识别和控车模拟闭环
- HaGRID 后续可用于训练静态手势分类器
- NVGesture 后续可作为动态手势识别扩展

============================================================
十三、交警手势数据集说明
============================================================

交警手势识别更适合使用全身姿态或交通警察指挥动作数据集。当前系统已基于 MediaPipe Pose 实现规则识别版本，可识别交警手势并输出交通指令。后续可结合专门交警手势数据集训练分类器。

============================================================
十四、Git 与上传说明
============================================================

推荐提交到 GitHub 的内容：

- 后端源码
- 前端源码
- requirements.txt
- package.json
- 训练脚本
- 文档
- demo 示例图片

不推荐提交的内容：

- backend/.venv/
- frontend/node_modules/
- backend/data/app.db
- backend/uploads/
- backend/outputs/
- backend/datasets/plate_sandbox/images/
- backend/datasets/plate_sandbox/annotations/
- backend/datasets/plate_sandbox_yolo/
- runs/
- .pt、.pkl、.onnx 等模型产物

常用命令：

git status
git add 文件或目录
git commit -m "提交说明"
git push origin main

============================================================
十五、当前阶段建议
============================================================

当前阶段建议优先完成：

1. 车主手势控车模块增强
   - 至少支持 6 种预定义手势
   - 支持图片和实时视频流输入
   - Web 页面展示手部关键点骨架
   - Web 页面展示车辆控制状态变化
   - 增加误触发抑制机制

2. 车牌训练暂缓
   - 等沙盘车移动后重新采集动态车牌图片
   - 再进行 LabelMe 标注和 YOLO 训练

3. 交警手势模块后续增强
   - 当前规则识别版本可用于演示
   - 后续有更多数据后再训练分类模型

============================================================
十六、常见问题
============================================================

1. 虚拟环境是干什么的？

虚拟环境用于给当前项目单独管理 Python 包，避免和系统 Python 或其他项目冲突。

2. 为什么 RTSP 视频流能读到帧，但识别不到车牌？

因为当前 HyperLPR3 模型对沙盘小车车牌不适配。需要后续采集沙盘动态数据并训练专门的车牌检测模型。

3. 为什么不要上传 datasets、uploads、outputs？

这些是运行数据、测试数据或训练数据，体积大且容易变化，不属于核心源码。

4. Swagger 文档页面白屏怎么办？

如果 /docs 白屏，可能是外部资源加载问题。可使用 /openapi.json 查看接口结构，不影响接口运行。

5. PowerShell 里中文乱码怎么办？

可以执行：

chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

============================================================
结束
============================================================
