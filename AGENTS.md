# 复现仓库约束

- 唯一执行配置为 `src/breakpoint_detection/baseline_settings.json`，固定现有验证最佳参数、阈值和checkpoint；不恢复网格搜索、不根据测试表现调参。
- 原始数据、参考源码、模型核心和已登记基准结果保持；新增运行写入新目录，不覆盖 `results/yahoo-a4`。
- 记录真实运行命令、时间、退出码、环境与原始stdout/stderr；不生成伪历史日志或回填提交时间。
- README使用中文，说明当前方法、环境、结果、运行方式与限制。新提交历史从实际导出开始，原目录没有可继承的Git历史。
