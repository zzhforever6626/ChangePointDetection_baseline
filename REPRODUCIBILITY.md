# 复现说明

## 环境安装

运行使用两个Python环境：

| 环境 | Python | 核心依赖 | 用途 |
|---|---|---|---|
| baseline | 3.10.21 | NumPy 2.2.6、pandas 2.3.3 | 数据处理、传统模型与评价 |
| tscp2 | 3.9.25 | NumPy 1.26.4、Torch 2.7.1+cu128 | TSCP2 checkpoint推理 |

在仓库根目录安装：

```powershell
conda create -n hkbu-baseline-repro python=3.10.21 pip
conda run -n hkbu-baseline-repro python -m pip install -r environments/requirements-test.txt
conda run -n hkbu-baseline-repro python -m pip install -r environments/requirements-build.txt
conda run -n hkbu-baseline-repro python -m pip install --no-deps --no-build-isolation -e .

conda create -n hkbu-tscp2-repro python=3.9.25 pip
conda run -n hkbu-tscp2-repro python -m pip install -r environments/requirements-tscp2.txt
conda run -n hkbu-tscp2-repro python -m pip install -r environments/requirements-torch-cu128.txt
conda run -n hkbu-tscp2-repro python -c "import sys; print(sys.executable)"
```

将最后输出的解释器路径填入 `--tscp2-python`。环境文件也提供 [baseline.yml](environments/baseline.yml) 和 [tscp2.yml](environments/tscp2.yml)。

记录中的执行平台为Windows x86_64，三个模型均使用CPU。NumPy依赖按运行时版本固定；具体版本及数值库构建信息见 `logs/baseline-verification/environment.json` 和 `provenance/environments`。

## 运行

```powershell
conda activate hkbu-baseline-repro
python -m pytest -q
python scripts/reproduce.py --output reproductions/my-run --tscp2-python PATH_TO_TORCH_PYTHON --workers 4
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
