# 正式baseline参考源码

来源文件及既有哈希原样保留；本说明按当前固定baseline更新。依据为2021论文明确适用要求、原方法及作者源码、登记的项目补充，唯一设置见 [baseline_settings.json](../baseline_settings.json)。

- RuLSIF：作者MATLAB提交 `496042e79b33c794cf2d6807496499763818508a`，MIT声明保留。当前λ=.01、demo α=.01、k10，按容差固定n25／25／10；比较边界s+n为项目映射。
- FLOSS：作者流式示例六个MATLAB文件及Python核心原样保留。当前W25／25／55、buffer800、原尺度异常替换后.001扰动、top-k3、闭区间±5W。
- TSCP2：固定作者与keras-tcn来源见 [manifest.json](tscp2/manifest.json)。当前W32、batch32、code16、lr.0003，训练更新BN统计、推理使用运行统计；固定385轮checkpoint。

原来源文件中默认参数不等于当前项目配置。未恢复完整作者Yahoo runner，不声称运行MATLAB二进制或逐位复现原作者训练。
