# 复现说明

## 环境安装

三模型共用一个Python 3.10环境：

| Python | NumPy | pandas | PyTorch | 运行设备 |
|---|---|---|---|---|
| 3.10.21 | 2.2.6 | 2.3.3 | 2.7.1+cpu | CPU |

在仓库根目录安装：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

`requirements.txt` 包含运行、测试和打包依赖，PyTorch使用CPU构建。

执行平台为Windows x86_64。具体版本及数值库构建信息见 `logs/baseline-verification/environment.json` 和 `provenance/environments/native-runtime.txt`。

## 运行

```powershell
python -m pytest -q
python scripts/reproduce.py --output reproductions/my-run --workers 4
python scripts/verify.py reproductions/my-run
```

每次使用新的输出目录。默认执行方式：

| 模型 | 评分方式 |
|---|---|
| FLOSS | 重算两个固定窗口，共120条验证／测试评分曲线 |
| RuLSIF | 读取120条固定窗口评分轨迹，重新生成峰、报警和指标 |
| TSCP2 | 复载固定checkpoint，对60条验证／测试序列推理 |

加 `--fresh-rulsif` 可重算全部RuLSIF密度评分。TSCP2使用385轮训练得到的checkpoint，训练日志和损失曲线位于 `provenance/original-tscp2-training`。

## 输出与指标

每个模型输出：

- `stdout.log`、`stderr.log`：进程输出。
- `process.json`：命令、起止时间、退出码和评分方式。
- `test-output.jsonl`：40条测试序列×3个容差，共120行完整测试输出。
- `run/<model>/metrics.csv`、`events.csv`、`summary.csv`：逐文件指标、报警和汇总。

`scripts/verify.py` 独立核验540行验证／测试指标和18行汇总，生成 `verification.json`、`metric-comparison.csv` 和 `metric-comparison.md`。指标表包含9组测试结果及对应论文参考F1。

完整运行记录位于 [logs/baseline-verification](logs/baseline-verification)，逐文件指标和报警保存在 `results/yahoo-a4`。

## 数据与配置

仓库包含100条Yahoo A4序列、模型源码、固定checkpoint和传统模型评分轨迹。统一配置为 `src/breakpoint_detection/baseline_settings.json`；`results/yahoo-a4/baseline-assets.json` 记录资产哈希。

数据划分、预处理、固定参数和事件评价规则见 [README](README.md)，数据与参考代码来源见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
