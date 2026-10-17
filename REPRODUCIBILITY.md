# 独立复现说明

## 仓库与提交记录

此目录含完整 `.git`，可直接 `git clone <此目录路径> 新目录`。准备阶段未连接GitHub远程；不以预计地址冒充已存在仓库。上传后以 `git remote get-url origin` 返回的真实链接作为提交材料中的独立复现链接。

原工作目录没有 `.git`，因此本仓库记录真实导入、复现工具、运行日志与最终核验的提交，不声称拥有原实验开发期的完整历史，不回填日期。查看全部实际记录：

```powershell
git log --all --format=fuller --stat
git fsck --full
```

## 环境

已验证平台为Windows x86_64、CPU评分与推理。基础解释器CPython 3.10.21、NumPy 2.2.6、pandas 2.3.3；独立神经解释器CPython 3.9.25、NumPy运行时1.26.4、Torch 2.7.1+cu128，实际使用CPU。保持两个环境，可避免把神经工作器的Python版本误认为包的最低版本。

从仓库根目录执行，使用新环境，不覆盖现有环境：

```powershell
conda create -n hkbu-baseline-repro python=3.10.21 pip
conda run -n hkbu-baseline-repro python -m pip install -r environments/requirements-test.txt
conda run -n hkbu-baseline-repro python -m pip install -r environments/requirements-build.txt
conda run -n hkbu-baseline-repro python -m pip install --no-deps --no-build-isolation -e .

conda create -n hkbu-tscp2-repro python=3.9.25 pip
conda run -n hkbu-tscp2-repro python -m pip install -r environments/requirements-tscp2.txt
conda run -n hkbu-tscp2-repro python -m pip install -r environments/requirements-torch-cu128.txt
conda run -n hkbu-tscp2-repro python -c "import sys; print(sys.executable)"
```

将最后输出的神经解释器路径填入下方命令。`torch==2.7.1+cu128` 的索引依据 [PyTorch官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。脚本固定CPU执行，不要求启用GPU；未验证换用CPU专用Torch构建、Linux或其他BLAS版本后的逐位一致性。

原神经环境的 `pip freeze` 中NumPy元数据显示1.21.6，但实际 `import numpy` 为1.26.4。依赖文件按实际运行版本登记，两个值同时保留在实际运行的 `environment.json`，不将原环境的过期元数据复制成安装要求。当前交付在既有实测环境运行验证，尚未从网络新建上述Conda环境验证安装过程。

## 运行与校验

```powershell
python -m pytest -q
python scripts/reproduce.py --output reproductions/my-run --tscp2-python PATH_TO_TORCH_PYTHON --workers 4
python scripts/verify.py reproductions/my-run
```

输出必须为新目录或空目录。脚本对三模型逐个实际启动独立进程，保留命令、UTC起止时间、退出码、原始stdout与stderr；同时输出全部40条测试序列×3个容差的完整预测、真值、计数和F1，每模型120行。环境和生成配置也一并保存。检验器独立执行最近距离事件匹配，逐行核验540行验证／测试指标、18行汇总和9组测试指标对照，额外核对每条预测与固定baseline一致。

默认模式：FLOSS重算两个选中窗口共120条验证／测试评分曲线；RuLSIF使用已打包且经过源码、参数、输入、轨迹哈希校验的120条选中窗口轨迹，再生成峰、报警和指标；TSCP2复载固定checkpoint并对60条验证／测试序列重新推理。完整重算RuLSIF可加 `--fresh-rulsif`，不重新选参。既有checkpoint的完整原始385轮训练stdout与训练曲线另保留在 `provenance/original-tscp2-training`；本次不重新训练。

默认模式可复现当前固定baseline，但不称从零训练三模型或重新拟合全部RuLSIF密度。旧评分生成阶段未保存的控制台日志不补造，本仓库原始测试日志来自本次实际执行，来源在每模型 `process.json` 中明示。

## 数据与资产

已使用本机已安装的setuptools后端离线构建wheel，通过记录见 `logs/build.stdout.log` 和 `logs/build-process.json`。一次没有输出而被终止的pip wheel尝试另保存在 `logs/initial-build`，不作为成功安装证据。wheel仅包含代码与包内参考文件；科学资产随完整Git仓库提供，因此日常复现采用仓库根目录的editable安装。

数据、checkpoint和固定传统模型轨迹均随仓库保存，无需访问原工作目录。唯一配置仍为 `baseline_settings.json`。评分资产清单是生成条件和完整性证据，不构成第二套默认配置。数据与参考源码的来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Git保留文件原字节，避免换行自动转换破坏源代码哈希。数据集切分、事件冲突处理与协议限制见README；测试已多次查看，不是新盲测，也不是论文原作者完整runner或论文数值的严格复现。

干净目录的首次pytest因旧临时目录父级不存在而报6个setup错误，原始记录保留在 `logs/initial-check`。导出仓库已将临时路径改为根目录下的 `.pytest-tmp`，随后30项测试通过；该修复只影响测试配置，不改变模型。

## 上传GitHub

维护者创建空仓库并提供真实地址后，可执行：

```powershell
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
git push origin --tags
```

当前目录已是独立Git仓库，无需重新 `git init` 或删除 `.git`。完整本地历史也可以通过旁边的Git bundle离线clone。原始数据与第三方代码的权利声明保持来源状态，未擅自补授许可。
