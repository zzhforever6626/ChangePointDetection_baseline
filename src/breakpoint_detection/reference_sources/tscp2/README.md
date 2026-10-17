# TSCP2来源与当前实现

作者仓库：https://github.com/cruiseresearchgroup/TSCP2 ，提交 `f353f09657f0ba73aec98d0cdeee0b8cc78fe9d3`。keras-tcn参考提交 `5282ac41027d4e42521d8d1cf877292166ea60c0`。原参考文件、许可证、URL和哈希见 [manifest.json](manifest.json)。

当前固定baseline保留2021论文两栈TCN、64通道、kernel4、dilations=[1,4,16]、三层投影BN、逐样本InfoNCE及前W个采样相似度均值的局部峰。PyTorch移植保留残差、skip和Flatten；投影宽度 `2*floor(W/2)→floor(W/2)→code`，输出BN；训练更新统计，推理使用运行统计。

固定设置为W32、batch32、code16、lr.0003、温度.1、训练／推理步长5、2W同文件样本分离。运行复载385轮训练得到的checkpoint，按固定阈值生成报警。
