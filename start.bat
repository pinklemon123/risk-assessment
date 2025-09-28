@echo off
chcp 65001 > nul
cls

echo ===========================================
echo    Mapping Editor 应用启动检查工具
echo ===========================================
echo.

:check_docker
echo 🔍 检查 Docker Desktop 状态...
docker system info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop 未运行
    echo.
    echo 📋 解决方案：
    echo   1. 启动 Docker Desktop 并等待
    echo   2. 使用本地模式运行（不需要 Docker）
    echo.
    echo 🤔 您想怎么做？
    echo   [1] 启动 Docker Desktop 并等待
    echo   [2] 使用本地模式运行
    echo   [3] 退出
    echo.
    set /p choice="请选择 (1-3): "
    
    if "%choice%"=="1" goto start_docker
    if "%choice%"=="2" goto local_mode
    if "%choice%"=="3" exit /b 0
    echo 无效选择，请重试
    goto check_docker
) else (
    echo ✅ Docker Desktop 正在运行
    goto docker_mode
)

:start_docker
echo � 正在启动 Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo ⏳ 等待 Docker Desktop 启动（这可能需要 1-2 分钟）...

:wait_docker
timeout /t 5 /nobreak >nul
docker system info >nul 2>&1
if errorlevel 1 (
    echo    还在启动中...
    goto wait_docker
)
echo ✅ Docker Desktop 启动完成！
goto docker_mode

:docker_mode
echo.
echo 🐳 使用 Docker 模式启动应用...
echo 📦 构建和启动容器...
docker-compose -f docker-compose.dev.yml up --build -d
if errorlevel 1 (
    echo ❌ Docker 启动失败，切换到本地模式
    goto local_mode
)
echo.
echo ✅ Docker 模式启动完成！
echo 🌐 访问地址：
echo   - 前端编辑器: http://localhost:3000
echo   - 后端 API: http://localhost:8080
echo   - API 文档: http://localhost:8080/docs
echo.
echo 📋 管理命令：
echo   docker-compose -f docker-compose.dev.yml logs -f  (查看日志)
echo   docker-compose -f docker-compose.dev.yml down     (停止服务)
goto end

:local_mode
echo.
echo 💻 使用本地模式启动应用...
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：Python 未安装或未添加到 PATH
    echo 请先安装 Python 3.8+ 并添加到 PATH
    pause
    exit /b 1
)

REM 检查 Node.js  
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：Node.js 未安装或未添加到 PATH
    echo 请先安装 Node.js 16+ 并添加到 PATH
    pause
    exit /b 1
)

echo ✅ Python 和 Node.js 检查通过

REM 安装依赖
echo 📦 检查并安装依赖...
pip install -r requirements.txt >nul 2>&1
if not exist node_modules (
    echo    安装 Node.js 依赖...
    npm install >nul 2>&1
)

echo 🚀 启动服务...

REM 启动后端
echo    启动后端服务 (端口 8080)...
start "" cmd /c "title=后端API服务 && cd /d %cd% && set PYTHONPATH=%cd% && python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"

REM 等待后端启动
timeout /t 3 /nobreak >nul

REM 启动前端
echo    启动前端服务 (端口 3000)...  
start "" cmd /c "title=前端编辑器 && cd /d %cd% && npm run dev"

echo.
echo ✅ 本地模式启动完成！
echo 🌐 访问地址：
echo   - 前端编辑器: http://localhost:3000
echo   - 后端 API: http://localhost:8080
echo   - API 文档: http://localhost:8080/docs
echo.
echo 💡 提示：
echo   - 已打开两个终端窗口分别运行前后端服务
echo   - 按 Ctrl+C 可停止对应服务
echo   - 关闭此窗口不会停止服务

:end
echo.
echo 🎉 启动完成！请稍等片刻让服务完全启动...
pause