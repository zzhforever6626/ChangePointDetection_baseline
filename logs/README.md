# 实验日志

[baseline-verification](baseline-verification) 保存FLOSS、RuLSIF和TSCP2的完整运行输出。

[tscp2-training](tscp2-training) 保存种子0从随机初始化训练的385轮记录：逐轮损失与相似度见 `training.csv`，原始输出见 `stdout.log`／`stderr.log`，训练命令、配置与环境见 `process.json`、`training-config.json`、`worker.json`，训练后核验见 `verification.json`。新权重的预测、事件计数和指标与[正式TSCP2结果](../results/yahoo-a4/tscp2)一致。

| 文件 | 内容 |
|---|---|
| 每模型 `stdout.log`、`stderr.log` | 原始进程输出 |
| 每模型 `process.json` | 命令、工作目录、起止时间、退出码和评分方式 |
| 每模型 `test-output.jsonl` | 120行测试输出，包含真值、全部预测和事件指标 |
| 每模型 `run/<model>/metrics.csv` | 验证／测试逐文件指标 |
| 每模型 `run/<model>/events.csv` | 完整报警记录 |
| 每模型 `run/<model>/summary.csv` | 验证／测试汇总 |
| 每模型 `run/run.json` | 输入、源码、输出哈希和运行设置 |

实验结果：[指标表](baseline-verification/metric-comparison.md)、[CSV](baseline-verification/metric-comparison.csv)、[核验报告](baseline-verification/verification.json)。三模型共360行测试输出、540行验证／测试指标和18行汇总，指标表列出9组测试结果与论文参考F1。

参数、种子与从头训练命令见 [环境与运行](../README.md#环境与运行)。

日志保留完整预测、报警和指标，评分中间数组默认在运行结束后清理。复现命令加 `--keep-scores` 可保留这些数组。
