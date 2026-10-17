# 实验日志

[baseline-verification](baseline-verification) 保存FLOSS、RuLSIF和TSCP2的完整运行输出。

[tscp2-training](tscp2-training) 保存种子0从随机初始化训练的385轮日志：`training.csv` 为逐轮损失与相似度，`stdout.log`／`stderr.log` 为原始输出，`process.json` 为实际训练命令，`training-config.json` 为训练配置，`worker.json` 为环境与训练汇总，`verification.json` 和 `weight-comparison.json` 为训练后核验；`metrics.csv`、`events.csv`、`summary.csv` 保存新权重的验证／测试结果。

| 文件 | 内容 |
|---|---|
| 每模型 `stdout.log`、`stderr.log` | 原始进程输出 |
| 每模型 `process.json` | 命令、工作目录、起止时间、退出码和评分方式 |
| 每模型 `test-output.jsonl` | 120行测试输出，包含真值、全部预测和事件指标 |
| 每模型 `run/<model>/metrics.csv` | 验证／测试逐文件指标 |
| 每模型 `run/<model>/events.csv` | 完整报警记录 |
| 每模型 `run/<model>/summary.csv` | 验证／测试汇总 |
| 每模型 `run/run.json` | 输入、源码、输出哈希和运行设置 |
| `pytest.stdout.log`、`pytest.stderr.log`、`pytest.xml` | 回归测试结果 |

实验结果：[指标表](baseline-verification/metric-comparison.md)、[CSV](baseline-verification/metric-comparison.csv)、[核验报告](baseline-verification/verification.json)。三模型共360行测试输出、540行验证／测试指标和18行汇总，指标表列出9组测试结果与论文参考F1。

默认运行重算FLOSS评分、读取RuLSIF评分轨迹并生成报警、复载TSCP2 checkpoint推理。模型参数与随机种子见 `src/breakpoint_detection/baseline_settings.json`。

从头训练与三模型完整运行命令见 [环境与运行](../README.md#环境与运行)。

日志保留完整预测、报警和指标，评分中间数组默认在运行结束后清理。复现命令加 `--keep-scores` 可保留这些数组。
