# 实验日志

## 运行实验

使用Python 3.10、Windows CPU环境，依赖版本固定在 [requirements.txt](../requirements.txt)。在仓库根目录执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe scripts/reproduce.py --output reproductions/full-run --workers 4 --fresh-rulsif --retrain-tscp2
```

完整流程包含FLOSS评分重算、RuLSIF全部密度估计重新拟合、TSCP2从随机初始化训练及结果核验；TSCP2在本机约需5小时。每次运行使用新的输出目录。

仅评估固定baseline（RuLSIF使用评分缓存、TSCP2复载已有checkpoint）：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce.py --output reproductions/evaluate --workers 4
```

两种流程结束后均自动核验指标。测试命令：`.\.venv\Scripts\python.exe -m pytest -q`。

## 实验输出

三模型运行输出与评价结果：

| 模型 | 原始日志与运行记录 | 完整测试输出 | 评价结果 |
|---|---|---|---|
| FLOSS | [stdout](baseline-verification/floss/stdout.log) · [stderr](baseline-verification/floss/stderr.log) · [运行记录](baseline-verification/floss/process.json) | [120行测试输出](baseline-verification/floss/test-output.jsonl) | [逐序列指标](baseline-verification/floss/run/floss/metrics.csv) · [报警](baseline-verification/floss/run/floss/events.csv) · [汇总](baseline-verification/floss/run/floss/summary.csv) |
| RuLSIF | [stdout](baseline-verification/rulsif/stdout.log) · [stderr](baseline-verification/rulsif/stderr.log) · [运行记录](baseline-verification/rulsif/process.json) | [120行测试输出](baseline-verification/rulsif/test-output.jsonl) | [逐序列指标](baseline-verification/rulsif/run/rulsif/metrics.csv) · [报警](baseline-verification/rulsif/run/rulsif/events.csv) · [汇总](baseline-verification/rulsif/run/rulsif/summary.csv) |
| TSCP2 | [stdout](baseline-verification/tscp2/stdout.log) · [stderr](baseline-verification/tscp2/stderr.log) · [运行记录](baseline-verification/tscp2/process.json) | [120行测试输出](baseline-verification/tscp2/test-output.jsonl) | [逐序列指标](baseline-verification/tscp2/run/tscp2/metrics.csv) · [报警](baseline-verification/tscp2/run/tscp2/events.csv) · [汇总](baseline-verification/tscp2/run/tscp2/summary.csv) |

每模型120行测试输出包含40条测试序列在3个容差下的真值、全部预测和事件指标。三模型共540行验证／测试指标、18行汇总。

实验结果：[指标对照表](baseline-verification/metric-comparison.md) · [对照CSV](baseline-verification/metric-comparison.csv) · [核验报告](baseline-verification/verification.json) · [运行环境](baseline-verification/environment.json)。

TSCP2从头训练（种子0，385轮）：[逐轮损失与相似度](tscp2-training/training.csv) · [stdout](tscp2-training/stdout.log) · [stderr](tscp2-training/stderr.log) · [训练命令](tscp2-training/process.json) · [训练配置](tscp2-training/training-config.json) · [环境与训练汇总](tscp2-training/worker.json) · [训练后核验](tscp2-training/verification.json)。新权重的预测和指标与[正式TSCP2汇总](../results/yahoo-a4/tscp2/summary.csv)一致。

参数与种子见[baseline_settings.json](../src/breakpoint_detection/baseline_settings.json)。评分中间数组默认在运行结束后清理，运行时加 `--keep-scores` 可保留。
