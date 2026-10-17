# Yahoo! A4 变点检测 Baseline

本项目实现 FLOSS、RuLSIF 和 TSCP2，包含独立运行所需的数据、模型代码、环境依赖、训练入口、checkpoint 和实验日志。全部设置见 [baseline_settings.json](src/breakpoint_detection/baseline_settings.json)，数据与算法来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 实验设置

使用100条 Yahoo A4 序列，每条1680点，共208个变点。按文件编号划分：TS1–40训练、TS41–60验证、TS61–100测试，对应86／43／79个变点。

| 模型 | 固定设置 |
|---|---|
| FLOSS | 容差24／50／75分别使用W=25／25／55；z-normalized Euclidean，buffer=800，top-k=3，闭区间±5W抑制，高斯扰动标准差0.001 |
| RuLSIF | 容差24／50／75分别使用n=W=25／25／10；embedding=10，α=0.01，λ=0.01，双向评分步长1，内部五折带宽交叉验证 |
| TSCP2 | W=32，batch=32，code=16，学习率0.0003；2个TCN栈、64通道、kernel=4、膨胀率[1,4,16]；InfoNCE温度0.1，训练／推理步长5 |

三个模型的随机种子均为0，FLOSS按种子、文件编号和窗口派生随机流。参数和阈值固定为验证集选定值，运行时不搜索；RuLSIF、TSCP2使用严格的 `score > threshold` 报警。

预处理插补缺失值，并按 `anomaly` 标签用相邻值替换异常点。FLOSS保留原尺度后加入扰动；RuLSIF和TSCP2对完整文件min-max归一化。RuLSIF以s+n为分界坐标；FLOSS中心评分覆盖400–1280，整文件结束后选点。整文件切分、n=W映射、FLOSS缓冲／随机流和TSCP2的PyTorch移植／BN设置均为本项目设置。

TSCP2训练时更新BN统计，推理时使用运行统计；同文件样本对起点至少相距2W。最多训练1600轮，第7轮起连续4轮损失上升或变化小于0.0001时早停，保留最后checkpoint。

评价采用事件micro-F1，容差为24／50／75，命中条件为 `abs(prediction-truth) < margin`。按距离、预测坐标、真值坐标依次执行一对一匹配；重复预测计FP，全部原始变点参与评价。跨文件累计TP／FP／FN后计算 `F1=2TP/(2TP+FP+FN)`。

## 环境与运行

运行环境为Windows x86_64、CPU、Python 3.10.21、NumPy 2.2.6、pandas 2.3.3、PyTorch 2.7.1+cpu，完整依赖固定在 [requirements.txt](requirements.txt)。在仓库根目录安装：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

完整实验：重新计算FLOSS评分、重新拟合RuLSIF全部密度估计，并从随机初始化训练TSCP2后核验新权重与指标。

```powershell
python scripts/reproduce.py --output reproductions/full-run --workers 4 --fresh-rulsif --retrain-tscp2
```

仅评估固定baseline：重新计算FLOSS评分、读取RuLSIF评分缓存、加载TSCP2 checkpoint推理。

```powershell
python scripts/reproduce.py --output reproductions/evaluate --workers 4
```

单独训练和核验TSCP2：

```powershell
python scripts/train_tscp2.py --output reproductions/tscp2-training
```

每次运行使用新的输出目录；`--keep-scores` 可保留评分中间数组。运行结束自动核验，也可执行 `python scripts/verify.py reproductions/full-run`；测试命令为 `python -m pytest -q`。

## 实验结果

| 模型 | 容差 | 验证F1 | 测试F1 |
|---|---:|---:|---:|
| FLOSS | 24 | 0.155340 | 0.140704 |
| FLOSS | 50 | 0.310680 | 0.251256 |
| FLOSS | 75 | 0.466019 | 0.311558 |
| RuLSIF | 24 | 0.216216 | 0.105263 |
| RuLSIF | 50 | 0.216216 | 0.105263 |
| RuLSIF | 75 | 0.228070 | 0.170492 |
| TSCP2 | 24 | 0.564103 | 0.464000 |
| TSCP2 | 50 | 0.564103 | 0.464000 |
| TSCP2 | 75 | 0.577778 | 0.529032 |

详细结果见 [FLOSS](results/yahoo-a4/floss/summary.csv)、[RuLSIF](results/yahoo-a4/rulsif/summary.csv)、[TSCP2](results/yahoo-a4/tscp2/summary.csv)，论文参考指标见 [指标对照](logs/baseline-verification/metric-comparison.md)。

原始运行输出见 [FLOSS日志](logs/baseline-verification/floss/stdout.log)、[RuLSIF日志](logs/baseline-verification/rulsif/stdout.log)、[TSCP2日志](logs/baseline-verification/tscp2/stdout.log)；训练过程见 [385轮训练记录](logs/tscp2-training/training.csv)及[训练核验结果](logs/tscp2-training/verification.json)。
