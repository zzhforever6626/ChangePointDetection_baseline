# Yahoo! A4 变点检测 Baseline

本项目实现 **FLOSS、RuLSIF、TSCP2** 三个 baseline，在公开 Yahoo A4 数据上采用统一事件评价协议。模型方法与明确设置依据 TS-CP² 2021 论文及作者公开源码；当前固定已完成验证搜索的最佳配置、阈值和 TSCP2 checkpoint。

唯一配置为 [baseline_settings.json](src/breakpoint_detection/baseline_settings.json)。正式运行直接评分和评价，不再执行参数网格、阈值搜索或神经网络训练。

本目录是可独立clone的Git仓库。GitHub远程地址尚未配置；实际提交历史从本次导出开始，原工作目录没有Git历史可继承。安装与复现步骤见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，原始运行日志和指标对照见 [日志说明](logs/README.md)。

## 当前结果

以下为固定配置的**事件 micro-F1**。验证集含20条序列、43个变点；测试集含40条序列、79个变点。

| 模型 | 容差 | 验证 F1 | 测试 F1 | 测试 TP / FP / FN | 固定设置 |
|---|---:|---:|---:|---|---|
| FLOSS | 24 | 0.155340 | 0.140704 | 14 / 106 / 65 | W=25 |
| FLOSS | 50 | 0.310680 | 0.251256 | 25 / 95 / 54 | W=25 |
| FLOSS | 75 | 0.466019 | 0.311558 | 31 / 89 / 48 | W=55 |
| RuLSIF | 24 | 0.216216 | 0.105263 | 8 / 65 / 71 | W=25 |
| RuLSIF | 50 | 0.216216 | 0.105263 | 8 / 65 / 71 | W=25 |
| RuLSIF | 75 | 0.228070 | 0.170492 | 26 / 200 / 53 | W=10 |
| TSCP2 | 24 | 0.564103 | 0.464000 | 29 / 17 / 50 | W=32, batch=32, code=16, lr=0.0003 |
| TSCP2 | 50 | 0.564103 | 0.464000 | 29 / 17 / 50 | W=32, batch=32, code=16, lr=0.0003 |
| TSCP2 | 75 | 0.577778 | 0.529032 | 41 / 35 / 38 | W=32, batch=32, code=16, lr=0.0003 |

固定阈值采用严格 `score > threshold`：

| 模型 | 容差24 | 容差50 | 容差75 |
|---|---:|---:|---:|
| RuLSIF | 34.97653253016169 | 34.97653253016169 | 52.3104435399189 |
| TSCP2 | 0.5519941318780184 | 0.5519941318780184 | 0.45870756125077605 |

FLOSS固定每文件top-k=3，无分数阈值。TSCP2固定使用已训练385轮的checkpoint；三个容差共享该模型，各自应用固定阈值。

固定配置已通过30项回归测试，540行逐文件指标与18行汇总独立复算一致。FLOSS的120条选中评分曲线实际重算；TSCP2复载checkpoint后评分逐位一致。RuLSIF复用经哈希核验的评分，并完成252次短序列方向拟合数值复核，本次未重新执行全量密度评分。

报警、逐文件计数和验证／测试汇总见 [FLOSS](results/yahoo-a4/floss/summary.csv)、[RuLSIF](results/yahoo-a4/rulsif/summary.csv)、[TSCP2](results/yahoo-a4/tscp2/summary.csv)。各模型的 `fixed-settings.json` 记录运行设置；[baseline-assets.json](results/yahoo-a4/baseline-assets.json)记录评分缓存的生成参数、输入和文件哈希。

## 模型实现与固定设置

- **RuLSIF**：作者相对密度比算法，λ=0.01，k=10、α=0.01；三个容差固定n=W=25／25／10。逐维样本标准差归一化、不减均值；保留算法内部五折带宽交叉验证、倍率[0.6,0.8,1,1.2,1.4]、最多100个分子核中心和未截断系数。完整双向扫描，步长1；事件坐标为比较边界s+n，另存绘图坐标s+2n+k−2。局部峰按固定阈值过滤，无额外NMS。
- **FLOSS**：作者流式核心，z-normalized Euclidean；三个容差固定W=25／25／55，buffer800、top-k=3、闭区间±5W抑制、seed0。异常点替换后保留原尺度，加入标准差0.001的高斯扰动；随机流由种子、文件编号、窗口派生。中心评分覆盖400–1280共881点，形成截止为坐标+400；整文件结束后选点。保留源码在剩余评分全为无穷时返回首坐标的行为。
- **TSCP2**：PyTorch实现，W=32、batch=32、code=16、lr=0.0003；2个TCN栈、64通道、kernel4、膨胀率[1,4,16]，保留残差、skip和时间／通道展平。三层投影宽度32→16→16，投影输出BN；训练时更新BN统计，推理使用运行统计。逐样本InfoNCE，温度0.1、训练／推理步长5、同文件样本对起点至少相距2W、种子0；前W个采样相似度作为移动历史。固定checkpoint来自最多1600轮、连续4轮损失上升或变化小于0.0001时停止的源码训练规则，保留最后checkpoint。

预处理线性插补缺失值，按 `anomaly` 标签用相邻值替换异常点；RuLSIF、TSCP2还对完整文件min-max归一化。采用完整文件离线评价，真值为原始 `changepoint` 标签。

## 数据与评价

使用 [公开Yahoo A4](data/A4Benchmark)全部100条序列，每条1680点，共168,000点、208个变点。按文件编号固定划分：TS1–40训练、TS41–60验证、TS61–100测试，变点数分别为86／43／79。

主指标为事件micro-F1，容差24、50、75均使用严格边界 `abs(prediction-truth) < margin`。一个真值最多匹配一个预测；额外预测计FP，包括同坐标重复预测；未命中真值计FN。冲突按距离、预测坐标、真值坐标排序，依次接受未匹配端点。跨文件累加TP／FP／FN后计算 `F1=2TP/(2TP+FP+FN)`；全部原始变点均参与，包括暖机和末端区域。

## 论文对照状态

当前模型遵循2021论文适用的明确方法设置。固定参数来自项目独立验证候选集，按事件micro-F1、precision、较少FP、候选ID及更高阈值依次解并列，未使用论文Table 2最佳值或测试F1作为选择目标。

文件划分、n=W映射、demo α、坐标与冲突处理、micro汇总、FLOSS短文件buffer和随机流、TSCP2移植及BN细节和验证阈值属于项目补充；作者精确Yahoo输入、测试索引及最优运行细节未确认。本项目不获取或重建作者初始输入。测试集已有多次查看，当前检验不属于新盲测，尚未复现论文数值或恢复完整作者runner。

## 运行

Python≥3.10，基础依赖NumPy、pandas；TSCP2推理需要PyTorch。

```powershell
python -m pip install -e ".[test,tscp2]"
python scripts/reproduce.py --output reproductions/my-run --tscp2-python PATH_TO_TORCH_PYTHON --workers 4
```

`--reuse-scores` 复用固定传统模型的评分轨迹，先校验生成参数、算法源码、输入和轨迹哈希，再重新生成报警与评价；省略此项则重新计算传统模型评分。TSCP2始终校验并复载固定checkpoint重新推理，不重新训练。checkpoint路径与SHA-256在唯一配置中登记，运行需要保留 `results/yahoo-a4/tscp2/encoder.pt`。

底层 `run-baseline` 可用 `--model floss`、`rulsif`、`tscp2` 单独评价，`--tscp2-python` 指定PyTorch解释器。`--workers` 控制传统模型评分进程数；TSCP2使用2个CPU线程。复现脚本默认实际重算FLOSS、复用经哈希核验的RuLSIF评分并重新推理TSCP2；`--fresh-rulsif` 可重新计算全部RuLSIF密度评分。

输出必须为新目录或空目录，不能覆盖正式结果。`--resume` 仅校验已完成运行的源码、输入及输出哈希；未完成运行须使用新目录。

```powershell
python -m pytest -q
```
