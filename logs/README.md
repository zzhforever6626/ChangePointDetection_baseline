# 实验日志

[baseline-verification](baseline-verification) 保存FLOSS、RuLSIF和TSCP2的完整运行输出。

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
| `build.stdout.log`、`build.stderr.log`、`build-process.json` | wheel构建记录 |

实验结果：[指标表](baseline-verification/metric-comparison.md)、[CSV](baseline-verification/metric-comparison.csv)、[核验报告](baseline-verification/verification.json)。三模型共360行测试输出、540行验证／测试指标和18行汇总，指标表列出9组测试结果与论文参考F1。

默认运行重算FLOSS评分、读取RuLSIF评分轨迹并生成报警、复载TSCP2 checkpoint推理。TSCP2的385轮训练日志位于 `provenance/original-tscp2-training`。
