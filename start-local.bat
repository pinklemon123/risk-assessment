@echo off
chcp 65001 > nul
cls

echo 🚀 启动 Mapping Editor 应用（本地模式）...
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装或未添加到 PATH
    echo 请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查 Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js 未安装或未添加到 PATH
    echo 请先安装 Node.js 16+
    pause
    exit /b 1
)

echo ✅ 环境检查通过
echo.

REM 安装 Python 依赖
echo 📦 安装 Python 依赖...
py -m pip install -r requirements.txt

echo.
echo 📦 安装 Node.js 依赖...
if not exist node_modules (
    npm install
) else (
    echo Node.js 依赖已存在
)

echo.
echo 🚀 启动服务...
echo.

REM 启动后端服务
echo 启动后端 API 服务 (端口 8080)...
start "Backend API" cmd /k "cd /d %cd% && set PYTHONPATH=%cd% && set API_KEYS=test_key && set ALLOWED_TENANTS=acme && set MAPPING_PATH=config/mapping.yaml && py -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"

REM 等待后端启动
echo 等待后端启动...
timeout /t 5 /nobreak > nul

REM 启动前端服务
echo 启动前端服务 (端口 3000)...
start "Frontend" cmd /k "cd /d %cd% && set API_URL=http://localhost:8080 && npm run dev"

timeout /t 2 /nobreak > nul

echo.
echo ✅ 应用启动完成！
echo.
echo 🌐 访问地址：
echo   - 前端编辑器: http://localhost:3000
echo   - 后端 API: http://localhost:8080  
echo   - API 文档: http://localhost:8080/docs
echo.
echo 💡 提示：
echo   - 两个命令窗口已打开，分别运行前后端服务
echo   - 按 Ctrl+C 可以停止对应的服务
echo   - 如需停止所有服务，请关闭对应的命令窗口
echo.
pause