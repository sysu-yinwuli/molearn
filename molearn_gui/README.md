# molearn_gui — Web 图形界面

**molearn_gui** 是 Molearn 项目的浏览器图形界面，基于 Flask 构建，支持 Windows / Linux / macOS。通过浏览器完成所有参数配置、步骤管理和模型训练，无需编辑任何配置文件。

> **独立性说明**：本 GUI 是独立组件，功能上不依赖 `molearn_run.py`；但在执行训练时，它会在后台调用 `molearn_run.py` 作为子进程。

---

## 安装

```bash
# 必要依赖（仅两个包）
pip install flask pyyaml

# 验证安装
python -c "import flask; print('Flask', flask.__version__)"
```

---

## 启动

```bash
# 标准启动（本机访问）
python molearn_gui/molearn_gui.py

# 指定端口
python molearn_gui/molearn_gui.py --port 8080

# 局域网共享（其他设备可通过 IP 访问）
python molearn_gui/molearn_gui.py --host 0.0.0.0 --port 5000

# 调试模式（开发用，自动重载）
python molearn_gui/molearn_gui.py --debug
```

启动后在浏览器打开 `http://localhost:5000` 即可使用。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址（`0.0.0.0` = 局域网可访问） |
| `--port` | `5000` | 监听端口 |
| `--debug` | 关 | Flask 调试模式 |

---

## 界面说明

### 侧边栏导航

GUI 采用单页应用设计，左侧侧边栏包含以下功能面板：

| 面板 | 功能 |
|------|------|
| **仪表盘** | 当前运行状态、快速操作按钮、实时日志输出 |
| **步骤管理** | 9 个步骤的启用/禁用切换，单步运行按钮 |
| **训练控制** | 种子设置、预演模式、一键训练/停止 |
| **输出文件** | 浏览所有输出文件，点击查看文件内容 |
| **项目信息** | 项目名称、描述、作者（写入 model_card.txt） |
| **目录路径** | 12 个数据/输出路径的统一配置 |
| **数据配置** | Step1（原始数据）/ Step4（抽样）/ Step5（划分）的参数 |
| **描述符** | 11 种描述符开关 + Morgan/SOAP/ACSF 详细参数 |
| **模型选择** | 16 种模型独立启用/禁用 |
| **特征开关** | 14 种特征标志（if_rdkit, if_morgan 等）|
| **HPO & 划分** | 超参数优化方法、CV、n_iter、SHAP、划分策略 |
| **降维配置** | PCA/KernelPCA/TSVD/UMAP/Autoencoder 参数 |
| **资源控制** | CPU 核数、内存上限、进程优先级 |

### 仪表盘

- **状态指示灯**：`idle`（空闲）/ `running`（运行中）/ `done`（完成）/ `error`（错误）
- **快速操作**：
  - **初始化目录**：自动创建全部 `data/` 和 `outputs/` 子目录
  - **运行全部步骤**：按 `molearn.yaml` 中 `enabled: true` 的步骤顺序执行
  - **停止运行**：立即终止当前运行的子进程
- **实时日志**：运行过程中日志逐行追加显示（Server-Sent Events，支持自动滚动）

### 实时日志技术

- **SSE（Server-Sent Events）**：默认使用，浏览器原生支持，低延迟
- **轮询模式（Fallback）**：SSE 不可用时自动切换，每 500ms 轮询一次 `/log_poll`
- 日志以文本行格式追加，最多显示最近 500 行

---

## API 接口

GUI 使用以下 REST API 与后端通信（开发者参考）：

| 方法 | 路由 | 功能 |
|------|------|------|
| `GET` | `/` | 返回主页 HTML |
| `GET` | `/get_config` | 获取当前 `molearn.yaml` 内容（JSON 格式） |
| `POST` | `/save_config` | 保存部分配置更新（深度合并） |
| `POST` | `/run` | 启动流水线（可选参数：`steps=3,6`） |
| `POST` | `/stop` | 终止当前运行 |
| `GET` | `/status` | 获取当前运行状态 `{status, log_len}` |
| `GET` | `/log_stream` | SSE 实时日志流 |
| `GET` | `/log_poll?offset=N` | 轮询日志（获取第 N 行之后的内容） |
| `GET` | `/outputs` | 列出所有输出文件 |
| `GET` | `/read_file?path=...` | 读取指定输出文件内容 |
| `POST` | `/init_dirs` | 创建项目目录结构 |

---

## 文件结构

```
molearn_gui/
├── molearn_gui.py          ← Flask 服务器（338 行）
├── templates/
│   └── index.html          ← 单页 Web UI（~1100 行 HTML/CSS/JS）
├── static/
│   ├── css/                ← 可选：自定义 CSS 文件
│   └── js/                 ← 可选：自定义 JS 文件
└── README.md               ← 本文件
```

---

## 注意事项

1. **并发限制**：同一时刻只允许运行一个流水线实例。点击"运行"前请确认当前无运行中任务。
2. **配置保存**：点击面板中的"保存"按钮后，修改立即写入 `molearn.yaml`；关闭浏览器不会丢失配置。
3. **PyYAML 警告**：未安装 `pyyaml` 时使用内置简易解析器，保存配置时**原有注释会丢失**。建议安装 `pip install pyyaml`。
4. **Windows 防火墙**：使用 `--host 0.0.0.0` 局域网共享时，Windows 可能弹出防火墙提示，需允许访问。
5. **端口冲突**：若 5000 端口被占用，使用 `--port 8080` 等其他端口。

---

## 常见问题

**Q: 启动时报 `ModuleNotFoundError: No module named 'flask'`**
```bash
pip install flask
```

**Q: 浏览器显示"无法连接"**

确认服务已启动（终端有 `Molearn GUI — http://127.0.0.1:5000` 输出），然后刷新浏览器。

**Q: 点击"保存"后配置未生效**

检查 `molearn.yaml` 文件是否存在于项目根目录。首次使用时确认文件已创建：
```bash
ls molearn.yaml   # 应存在
```

**Q: 日志不更新**

检查浏览器控制台（F12）是否有 EventSource 错误。如有，SSE 已自动切换到轮询模式，日志会稍有延迟（约 500ms）。

**Q: Windows 下中文乱码**

在 PowerShell 中先执行：
```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
python molearn_gui/molearn_gui.py
```
