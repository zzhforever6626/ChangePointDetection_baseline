# 原始实验日志

`baseline-verification/` 保存本次在独立导出仓库中实际执行的三模型固定配置运行；不把后处理输出伪装成历史未保存的stdout。

- 每模型 `stdout.log`、`stderr.log`：子进程原始输出，空stderr表示该进程没有写入错误流。
- 每模型 `process.json`：实际命令、工作目录、UTC起止时间、退出码及评分模式。
- 每模型 `test-output.jsonl`：本次运行产生的完整测试逐文件输出，每模型120行，含真值、全部预测和事件指标。
- 每模型 `run/<model>/metrics.csv`、`events.csv`、`summary.csv`：完整验证／测试逐文件指标、报警及汇总；`run/run.json` 保存源码、输入、输出哈希和固定设置。
- [metric-comparison.md](baseline-verification/metric-comparison.md)、[metric-comparison.csv](baseline-verification/metric-comparison.csv)：对照当前baseline的9组测试F1和项目登记的论文参考值。
- [verification.json](baseline-verification/verification.json)：独立逐文件匹配和汇总核验报告。
- `pytest.stdout.log`、`pytest.stderr.log`、`pytest.xml`：本导出仓库的真实测试记录。

运行的默认模式为FLOSS重算、RuLSIF核验评分回放及重新报警、TSCP2固定checkpoint复载推理。RuLSIF全量密度评分及神经训练未在此次日志中重新执行；TSCP2原始训练日志单独保留在 `provenance/original-tscp2-training`。
