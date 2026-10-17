# 正式baseline参考源码

本目录保存三模型的参考源码及文件哈希。当前运行设置见 [baseline_settings.json](../baseline_settings.json)。

- RuLSIF：作者MATLAB提交 `496042e79b33c794cf2d6807496499763818508a`，MIT声明保留。当前λ=.01、demo α=.01、k10，按容差固定n25／25／10；比较边界s+n为项目映射。
- FLOSS：作者流式示例六个MATLAB文件及Python核心原样保留。当前W25／25／55、buffer800、原尺度异常替换后.001扰动、top-k3、闭区间±5W。
- TSCP2：固定作者与keras-tcn来源见 [manifest.json](tscp2/manifest.json)。当前W32、batch32、code16、lr.0003，训练更新BN统计、推理使用运行统计；固定385轮checkpoint。

文件来源与许可见仓库根目录的 `THIRD_PARTY_NOTICES.md`。
