# TSCP2 公开来源与实现边界

作者仓库：[cruiseresearchgroup/TSCP2](https://github.com/cruiseresearchgroup/TSCP2)，commit `f353f09657f0ba73aec98d0cdeee0b8cc78fe9d3`。参考文件保持原内容，URL、SHA256、Git blob SHA1见 [manifest.json](manifest.json)。当前DataHelper未接通Yahoo；保留的Yahoo helper不能证明完整原实验runner已恢复。

TCN来源：[keras-tcn](https://github.com/philipperemy/keras-tcn)，commit `5282ac41027d4e42521d8d1cf877292166ea60c0`，为论文之前的依赖补充，作者实际版本未公开，保留MIT许可。

依据顺序为2024→2021→作者代码→必要项目适配。2024未重定义TSCP2结构；Appendix C的TS2Vec/BYOL训练参数不转移给TSCP2。

- 2021 Figure 3/§3.3的两栈、64 filters、kernel=4、dilations=[1,4,16]保持；论文膨胀率覆盖仓库默认[1,2,4,8]。
- 2021式(1)/(4)的逐样本InfoNCE覆盖代码全批正样本求和及BCE-logits；§3.4/Figure 7前W个相似度均值减当前值及局部峰覆盖直接相似度阈值。
- 保留残差、skip求和、时间/通道Flatten、三层Dense宽度 `2*floor(W/2)→floor(W/2)→10`。论文要求投影头BN，采用作者注释中最后Dense后的BN位置；原实验精确位置未公开。
- 卷积random_normal(0,.05)、Dense Glorot uniform、BN eps=.001/momentum=.99保持。作者手工调用model(x)无training=True或全局学习阶段设置；按TensorFlow2.2默认阶段0处理，统计量不更新，gamma/beta仍训练。此行为依据公开代码，不声称原实验实际环境必然如此。
- Adam采用TensorFlow2.2 epsilon-hat数学更新，Torch epsilon按 `1e-7/sqrt(1-beta2^t)` 缩放；beta=(.9,.999)、lr=1e-4。PyTorch移植不保证逐位训练一致。
- batch=64、code=10、temperature=.1、最大1600 epoch、训练损失四次停滞早停并保留最后模型来自源码默认；Yahoo最优batch/code未公开，不按测试成绩搜索。
- 2021 Table 2的最佳窗口是搜索结果，不是必须按容差固定的参数。当前正式流程在验证集比较W24/50/75，三容差实际均选W50；未覆盖W100及完整batch/code/lr网格。
- Yahoo helper训练步长5及严格末端范围保持；推理比较网格5/1，验证实际选择5。前W个相似度仍指前W个采样窗口。训练不跨文件、不按变点标签筛样本；同文件负样本起点至少2W为论文分离要求的明确适配。
- 作者先汇集窗口再切分，比例采用2024的40/20/40。等长100条A4使分界落在40/20/40整文件边界；数字排序为项目补充，未恢复作者顺序。
- 验证集选择统一报警间距160及各容差全局阈值，当前三个阈值均为0.2824683201。分数优先抑制、候选间距与阈值选择细则为项目适配，未称作者原Yahoo设置。局部峰确认截止c+W+5，最终批量抑制选点截止评价阶段结束。全部原始真值保留，包括缺完整评分的边缘变点。

未声称恢复完整原实验或复现论文数值。
