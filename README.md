# xiaoxueqi
暑期实践项目开发
小学期项目环境配置说明

本项目目前涉及 Java 后端、Python 算法服务、前端页面、RTSP 视频流、FFmpeg 推流、MediaMTX 流媒体服务器、VLC 拉流测试、GitHub 协作。

为了保证小组成员本地环境一致，建议按照下面的软件清单配置。

============================================================
一、项目仓库
============================================================

GitHub 仓库地址：

https://github.com/babyxxrz/xiaoxueqi.git

本地建议克隆路径：

D:\code\xiaoxueqi

拉取方式：

cd D:\code
git clone https://github.com/babyxxrz/xiaoxueqi.git
cd xiaoxueqi


============================================================
二、必须安装的软件和环境
============================================================

1. VS Code

用途：
- 主要开发工具
- 编写 Java、Python、前端代码
- 打开 Git 仓库
- 运行终端命令

建议安装插件：

- Chinese (Simplified) Language Pack
- Extension Pack for Java
- Spring Boot Extension Pack
- Python
- REST Client

插件说明：

- Extension Pack for Java：Java 开发基础插件包
- Spring Boot Extension Pack：Spring Boot 后端开发
- Python：Python 算法开发
- REST Client：在 VS Code 里测试接口，可以替代 Apifox / Postman


------------------------------------------------------------
2. JDK 17
------------------------------------------------------------

用途：
- 运行 Java 后端项目
- 编译 Java 代码
- 支持 Spring Boot 项目

推荐版本：

JDK 17

当前已验证版本：

OpenJDK 17.0.9 Temurin

检查命令：

java -version
javac -version

正常结果应类似：

openjdk version "17.0.9"
javac 17.0.9


------------------------------------------------------------
3. Maven
------------------------------------------------------------

用途：
- Java 后端依赖管理
- Spring Boot 项目构建

当前已验证版本：

Apache Maven 3.9.15

当前本机 Maven 路径：

D:\apache-maven-3.9.15

检查命令：

mvn -version


------------------------------------------------------------
4. Git
------------------------------------------------------------

用途：
- 拉取小组仓库
- 提交代码
- 推送代码到 GitHub

当前已验证版本：

git version 2.55.0.windows.2

检查命令：

git --version

首次使用需要配置：

git config --global user.name "你的GitHub用户名"
git config --global user.email "你的邮箱"
git config --global init.defaultBranch main


------------------------------------------------------------
5. Node.js 和 npm
------------------------------------------------------------

用途：
- 后续前端项目开发
- 运行 Vue / Vite 项目
- 管理前端依赖

当前已验证版本：

Node.js v24.16.0
npm 11.13.0

检查命令：

node -v
npm -v


------------------------------------------------------------
6. Python 3.11
------------------------------------------------------------

用途：
- Python 算法服务
- OpenCV 读取 RTSP 视频流
- 后续 YOLO / MediaPipe 识别功能

推荐版本：

Python 3.11.x

当前已验证版本：

Python 3.11.9
pip 26.1.2

不建议安装过新的 Python 版本，例如 Python 3.13 / 3.14，因为 OpenCV、MediaPipe、YOLO 等视觉相关库可能出现兼容问题。

检查命令：

python --version
pip --version
py --version


------------------------------------------------------------
7. FFmpeg
------------------------------------------------------------

用途：
- 把本地视频或摄像头视频推流到 MediaMTX
- 生成 RTSP 视频流
- 后续可能用于截图、转码、保存检测片段

老师提供的 FFmpeg 下载地址：

https://github.com/BtbN/FFmpeg-Builds/releases

当前本机使用的是 winget 安装的 Gyan 版本，也可以正常使用：

ffmpeg version 8.1.2-full_build-www.gyan.dev
ffprobe version 8.1.2-full_build-www.gyan.dev

检查命令：

ffmpeg -version
ffprobe -version


------------------------------------------------------------
8. MediaMTX
------------------------------------------------------------

用途：
- 本地流媒体服务器
- 接收 FFmpeg 推流
- 向 VLC、Python OpenCV、前端或后端提供 RTSP 视频流地址

老师提供的 MediaMTX 地址：

https://github.com/bluenviron/mediamtx

需要下载 Windows 64 位版本，文件名类似：

mediamtx_v1.19.2_windows_amd64.zip

建议解压路径：

D:\tools\mediamtx

目录里应包含：

mediamtx.exe
mediamtx.yml

启动命令：

cd D:\tools\mediamtx
.\mediamtx.exe

注意：
运行 mediamtx.exe 的 PowerShell 窗口不能关闭，关闭后流媒体服务器会停止。


------------------------------------------------------------
9. VLC
------------------------------------------------------------

用途：
- 测试 RTSP 视频流是否可以正常播放
- 判断视频流本身是否正常，避免一开始就怀疑 Python 或后端代码

使用方式：

VLC → 媒体 → 打开网络串流

测试地址示例：

rtsp://localhost:8554/test


============================================================
三、可选软件
============================================================

1. MySQL

用途：
- 保存用户信息
- 保存车牌识别记录
- 保存手势识别记录
- 保存日志和告警数据

本机目前已有 MySQL，所以暂时跳过重新安装。

建议版本：

MySQL 8.x 或更高版本

检查命令：

mysql --version

后续 Spring Boot 连接数据库时，需要统一：

- 数据库地址
- 端口
- 用户名
- 密码
- 数据库名称


------------------------------------------------------------
2. DBeaver / MySQL Workbench
------------------------------------------------------------

用途：
- 图形化查看数据库
- 建库建表
- 查询识别记录

DBeaver 和 MySQL Workbench 二者任选一个即可。


------------------------------------------------------------
3. Apifox / Postman
------------------------------------------------------------

用途：
- 测试后端接口
- 测试 Python 算法接口
- 保存接口请求

目前本项目可以先不安装 Apifox，使用 VS Code 插件 REST Client 替代。


============================================================
四、Python 算法环境配置
============================================================

Python 虚拟环境不要上传 GitHub，每个人本地自己创建。

进入算法目录：

cd D:\code\xiaoxueqi\algorithm

创建虚拟环境：

python -m venv .venv

激活虚拟环境：

.\.venv\Scripts\activate

激活后命令行前面会出现：

(.venv)

安装依赖：

pip install -r requirements.txt

当前已经上传的依赖文件：

algorithm/requirements.txt

如果后续安装了新的 Python 包，例如：

fastapi
uvicorn
ultralytics
mediapipe

需要重新生成依赖文件：

pip freeze > requirements.txt

然后提交到 GitHub。


============================================================
五、当前已经验证成功的视频流流程
============================================================

目前已经验证成功的链路：

本地视频
  ↓
FFmpeg 推流
  ↓
MediaMTX 流媒体服务器
  ↓
VLC 拉流播放
  ↓
Python OpenCV 读取 RTSP 视频流

测试 RTSP 地址：

rtsp://localhost:8554/test


============================================================
六、MediaMTX + FFmpeg + VLC 测试流程
============================================================

1. 启动 MediaMTX

打开一个 PowerShell：

cd D:\tools\mediamtx
.\mediamtx.exe

保持这个窗口运行。


------------------------------------------------------------
2. 使用 FFmpeg 推流
------------------------------------------------------------

另外打开一个 PowerShell。

假设测试视频路径为：

D:\test.mp4

执行：

ffmpeg -re -stream_loop -1 -i D:\test.mp4 -c copy -f rtsp rtsp://localhost:8554/test

如果 -c copy 出现卡顿或 H.264 解码错误，可以改用重新编码方式：

ffmpeg -re -stream_loop -1 -i D:\test.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an -f rtsp -rtsp_transport tcp rtsp://localhost:8554/test


------------------------------------------------------------
3. 使用 VLC 拉流
------------------------------------------------------------

VLC 中打开网络串流：

rtsp://localhost:8554/test

如果能播放，说明 FFmpeg 推流和 MediaMTX 服务正常。


------------------------------------------------------------
4. 使用 Python 读取 RTSP
------------------------------------------------------------

进入算法目录：

cd D:\code\xiaoxueqi\algorithm
.\.venv\Scripts\activate

运行：

python rtsp_test.py

如果弹出视频窗口，说明 Python OpenCV 可以读取 RTSP 视频流。

退出方式：

按 q


============================================================
七、当前已上传到 GitHub 的文件
============================================================

目前已经上传到 GitHub 的环境/测试相关文件：

.gitignore
algorithm/requirements.txt
algorithm/rtsp_test.py

其中：

.gitignore：防止上传 .venv、node_modules、视频文件、日志等本地文件
requirements.txt：Python 依赖清单
rtsp_test.py：Python 读取 RTSP 视频流的测试脚本


============================================================
八、不要提交到 GitHub 的内容
============================================================

以下内容不要上传：

algorithm/.venv/
node_modules/
target/
dist/
*.mp4
*.avi
*.mov
*.mkv
*.log
.env

原因：

.venv 是本地 Python 虚拟环境
node_modules 是前端依赖
target 是 Java 编译产物
视频文件体积较大
.env 可能包含密钥或本地配置
日志文件没有必要上传


============================================================
九、推荐 .gitignore 内容
============================================================

# Python
algorithm/.venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Node
node_modules/
dist/

# Java
target/
*.class

# Media / test files
*.mp4
*.avi
*.mov
*.mkv

# Env
.env

# Logs
*.log

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db


============================================================
十、后续还需要开发的内容
============================================================

当前环境链路已经打通，后续可以继续做：

1. Python FastAPI 算法服务
2. RTSP 视频流截图接口
3. YOLO 车辆 / 车牌检测
4. MediaPipe 手势识别
5. Java Spring Boot 后端
6. 前端页面
7. MySQL 数据库连接
8. 识别结果保存
9. 告警日志模块
10. 前后端联调


============================================================
十一、给 AI 的补充说明
============================================================

本项目已经完成以下环境验证：

VS Code 已安装
JDK 17 已配置
Maven 3.9.15 已配置
Git 已配置并成功 push 到 GitHub
Node.js / npm 已配置
Python 3.11.9 已配置
pip 已升级
FFmpeg 已安装并验证
MediaMTX 已启动并测试
VLC 已能播放 RTSP 流
Python OpenCV 已能读取 rtsp://localhost:8554/test

项目当前重点不是重新配置基础环境，而是在现有基础上继续开发：

algorithm/ 下的 Python 算法服务
backend/ 下的 Java 后端
frontend/ 下的前端页面
视频流识别与展示逻辑

其他成员拉取仓库后，需要自己安装软件和创建虚拟环境；GitHub 只同步代码和依赖清单，不会自动同步 VS Code 插件、Python 本体、.venv、FFmpeg、MediaMTX、VLC 等本地环境。
=======
# xiaoxueqi
暑期实践项目开发
小学期项目环境配置说明

本项目目前涉及 Java 后端、Python 算法服务、前端页面、RTSP 视频流、FFmpeg 推流、MediaMTX 流媒体服务器、VLC 拉流测试、GitHub 协作。

为了保证小组成员本地环境一致，建议按照下面的软件清单配置。

============================================================
一、项目仓库
============================================================

GitHub 仓库地址：

https://github.com/babyxxrz/xiaoxueqi.git

本地建议克隆路径：

D:\code\xiaoxueqi

拉取方式：

cd D:\code
git clone https://github.com/babyxxrz/xiaoxueqi.git
cd xiaoxueqi


============================================================
二、必须安装的软件和环境
============================================================

1. VS Code

用途：
- 主要开发工具
- 编写 Java、Python、前端代码
- 打开 Git 仓库
- 运行终端命令

建议安装插件：

- Chinese (Simplified) Language Pack
- Extension Pack for Java
- Spring Boot Extension Pack
- Python
- REST Client

插件说明：

- Extension Pack for Java：Java 开发基础插件包
- Spring Boot Extension Pack：Spring Boot 后端开发
- Python：Python 算法开发
- REST Client：在 VS Code 里测试接口，可以替代 Apifox / Postman


------------------------------------------------------------
2. JDK 17
------------------------------------------------------------

用途：
- 运行 Java 后端项目
- 编译 Java 代码
- 支持 Spring Boot 项目

推荐版本：

JDK 17

当前已验证版本：

OpenJDK 17.0.9 Temurin

检查命令：

java -version
javac -version

正常结果应类似：

openjdk version "17.0.9"
javac 17.0.9


------------------------------------------------------------
3. Maven
------------------------------------------------------------

用途：
- Java 后端依赖管理
- Spring Boot 项目构建

当前已验证版本：

Apache Maven 3.9.15

当前本机 Maven 路径：

D:\apache-maven-3.9.15

检查命令：

mvn -version


------------------------------------------------------------
4. Git
------------------------------------------------------------

用途：
- 拉取小组仓库
- 提交代码
- 推送代码到 GitHub

当前已验证版本：

git version 2.55.0.windows.2

检查命令：

git --version

首次使用需要配置：

git config --global user.name "你的GitHub用户名"
git config --global user.email "你的邮箱"
git config --global init.defaultBranch main


------------------------------------------------------------
5. Node.js 和 npm
------------------------------------------------------------

用途：
- 后续前端项目开发
- 运行 Vue / Vite 项目
- 管理前端依赖

当前已验证版本：

Node.js v24.16.0
npm 11.13.0

检查命令：

node -v
npm -v


------------------------------------------------------------
6. Python 3.11
------------------------------------------------------------

用途：
- Python 算法服务
- OpenCV 读取 RTSP 视频流
- 后续 YOLO / MediaPipe 识别功能

推荐版本：

Python 3.11.x

当前已验证版本：

Python 3.11.9
pip 26.1.2

不建议安装过新的 Python 版本，例如 Python 3.13 / 3.14，因为 OpenCV、MediaPipe、YOLO 等视觉相关库可能出现兼容问题。

检查命令：

python --version
pip --version
py --version


------------------------------------------------------------
7. FFmpeg
------------------------------------------------------------

用途：
- 把本地视频或摄像头视频推流到 MediaMTX
- 生成 RTSP 视频流
- 后续可能用于截图、转码、保存检测片段

老师提供的 FFmpeg 下载地址：

https://github.com/BtbN/FFmpeg-Builds/releases

当前本机使用的是 winget 安装的 Gyan 版本，也可以正常使用：

ffmpeg version 8.1.2-full_build-www.gyan.dev
ffprobe version 8.1.2-full_build-www.gyan.dev

检查命令：

ffmpeg -version
ffprobe -version


------------------------------------------------------------
8. MediaMTX
------------------------------------------------------------

用途：
- 本地流媒体服务器
- 接收 FFmpeg 推流
- 向 VLC、Python OpenCV、前端或后端提供 RTSP 视频流地址

老师提供的 MediaMTX 地址：

https://github.com/bluenviron/mediamtx

需要下载 Windows 64 位版本，文件名类似：

mediamtx_v1.19.2_windows_amd64.zip

建议解压路径：

D:\tools\mediamtx

目录里应包含：

mediamtx.exe
mediamtx.yml

启动命令：

cd D:\tools\mediamtx
.\mediamtx.exe

注意：
运行 mediamtx.exe 的 PowerShell 窗口不能关闭，关闭后流媒体服务器会停止。


------------------------------------------------------------
9. VLC
------------------------------------------------------------

用途：
- 测试 RTSP 视频流是否可以正常播放
- 判断视频流本身是否正常，避免一开始就怀疑 Python 或后端代码

使用方式：

VLC → 媒体 → 打开网络串流

测试地址示例：

rtsp://localhost:8554/test


============================================================
三、可选软件
============================================================

1. MySQL

用途：
- 保存用户信息
- 保存车牌识别记录
- 保存手势识别记录
- 保存日志和告警数据

本机目前已有 MySQL，所以暂时跳过重新安装。

建议版本：

MySQL 8.x 或更高版本

检查命令：

mysql --version

后续 Spring Boot 连接数据库时，需要统一：

- 数据库地址
- 端口
- 用户名
- 密码
- 数据库名称


------------------------------------------------------------
2. DBeaver / MySQL Workbench
------------------------------------------------------------

用途：
- 图形化查看数据库
- 建库建表
- 查询识别记录

DBeaver 和 MySQL Workbench 二者任选一个即可。


------------------------------------------------------------
3. Apifox / Postman
------------------------------------------------------------

用途：
- 测试后端接口
- 测试 Python 算法接口
- 保存接口请求

目前本项目可以先不安装 Apifox，使用 VS Code 插件 REST Client 替代。


============================================================
四、Python 算法环境配置
============================================================

Python 虚拟环境不要上传 GitHub，每个人本地自己创建。

进入算法目录：

cd D:\code\xiaoxueqi\algorithm

创建虚拟环境：

python -m venv .venv

激活虚拟环境：

.\.venv\Scripts\activate

激活后命令行前面会出现：

(.venv)

安装依赖：

pip install -r requirements.txt

当前已经上传的依赖文件：

algorithm/requirements.txt

如果后续安装了新的 Python 包，例如：

fastapi
uvicorn
ultralytics
mediapipe

需要重新生成依赖文件：

pip freeze > requirements.txt

然后提交到 GitHub。


============================================================
五、当前已经验证成功的视频流流程
============================================================

目前已经验证成功的链路：

本地视频
  ↓
FFmpeg 推流
  ↓
MediaMTX 流媒体服务器
  ↓
VLC 拉流播放
  ↓
Python OpenCV 读取 RTSP 视频流

测试 RTSP 地址：

rtsp://localhost:8554/test


============================================================
六、MediaMTX + FFmpeg + VLC 测试流程
============================================================

1. 启动 MediaMTX

打开一个 PowerShell：

cd D:\tools\mediamtx
.\mediamtx.exe

保持这个窗口运行。


------------------------------------------------------------
2. 使用 FFmpeg 推流
------------------------------------------------------------

另外打开一个 PowerShell。

假设测试视频路径为：

D:\test.mp4

执行：

ffmpeg -re -stream_loop -1 -i D:\test.mp4 -c copy -f rtsp rtsp://localhost:8554/test

如果 -c copy 出现卡顿或 H.264 解码错误，可以改用重新编码方式：

ffmpeg -re -stream_loop -1 -i D:\test.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an -f rtsp -rtsp_transport tcp rtsp://localhost:8554/test


------------------------------------------------------------
3. 使用 VLC 拉流
------------------------------------------------------------

VLC 中打开网络串流：

rtsp://localhost:8554/test

如果能播放，说明 FFmpeg 推流和 MediaMTX 服务正常。


------------------------------------------------------------
4. 使用 Python 读取 RTSP
------------------------------------------------------------

进入算法目录：

cd D:\code\xiaoxueqi\algorithm
.\.venv\Scripts\activate

运行：

python rtsp_test.py

如果弹出视频窗口，说明 Python OpenCV 可以读取 RTSP 视频流。

退出方式：

按 q


============================================================
七、当前已上传到 GitHub 的文件
============================================================

目前已经上传到 GitHub 的环境/测试相关文件：

.gitignore
algorithm/requirements.txt
algorithm/rtsp_test.py

其中：

.gitignore：防止上传 .venv、node_modules、视频文件、日志等本地文件
requirements.txt：Python 依赖清单
rtsp_test.py：Python 读取 RTSP 视频流的测试脚本


============================================================
八、不要提交到 GitHub 的内容
============================================================

以下内容不要上传：

algorithm/.venv/
node_modules/
target/
dist/
*.mp4
*.avi
*.mov
*.mkv
*.log
.env

原因：

.venv 是本地 Python 虚拟环境
node_modules 是前端依赖
target 是 Java 编译产物
视频文件体积较大
.env 可能包含密钥或本地配置
日志文件没有必要上传


============================================================
九、推荐 .gitignore 内容
============================================================

# Python
algorithm/.venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Node
node_modules/
dist/

# Java
target/
*.class

# Media / test files
*.mp4
*.avi
*.mov
*.mkv

# Env
.env

# Logs
*.log

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db


============================================================
十、后续还需要开发的内容
============================================================

当前环境链路已经打通，后续可以继续做：

1. Python FastAPI 算法服务
2. RTSP 视频流截图接口
3. YOLO 车辆 / 车牌检测
4. MediaPipe 手势识别
5. Java Spring Boot 后端
6. 前端页面
7. MySQL 数据库连接
8. 识别结果保存
9. 告警日志模块
10. 前后端联调


============================================================
十一、给 AI 的补充说明
============================================================

本项目已经完成以下环境验证：

VS Code 已安装
JDK 17 已配置
Maven 3.9.15 已配置
Git 已配置并成功 push 到 GitHub
Node.js / npm 已配置
Python 3.11.9 已配置
pip 已升级
FFmpeg 已安装并验证
MediaMTX 已启动并测试
VLC 已能播放 RTSP 流
Python OpenCV 已能读取 rtsp://localhost:8554/test

项目当前重点不是重新配置基础环境，而是在现有基础上继续开发：

algorithm/ 下的 Python 算法服务
backend/ 下的 Java 后端
frontend/ 下的前端页面
视频流识别与展示逻辑

其他成员拉取仓库后，需要自己安装软件和创建虚拟环境；GitHub 只同步代码和依赖清单，不会自动同步 VS Code 插件、Python 本体、.venv、FFmpeg、MediaMTX、VLC 等本地环境。
