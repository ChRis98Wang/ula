# 外部情绪身体动作 / 动捕数据集候选审计

审计日期：2026-07-27
范围：只调研 XEM、BEAT2 之外的情绪身体动作或动捕数据集。
操作边界：只读取官方项目页、作者仓库、机构数据仓库和原始论文；为了核验标签质量，
仅把 Hanyang 的 743.5 kB 人类评价表下载到系统临时目录做聚合统计，没有下载动作包，
也没有把任何外部数据放入项目、缓存或训练。

## 结论

可以从“情绪数据”下手，而且比继续把弱情绪标签直接塞进基础生成器更可靠。但是当前
公开数据只能支持**情绪表征、情绪判别器、retarget 适配器和小规模 A/B**，尚不足以
单独替代 BEAT2，训练出同时满足“60 秒共语、多人泛化、18D 机器人可执行、情绪准确”
的最终生成器：

- **科学质量第一：DIEM-A。** 97 位专业表演者、12 种情绪加中性、三档强度、三语
  场景文本，且提供 24 关节 BVH；这是最接近目标的外部情绪动作库。但必须由合格 PI
  签署 ULA，只限非商业学术研究，不能直接用于生产或公开权重。
- **当前最可执行：Hanyang/Duksung Emotional Body Motion。** 4,060 段、29 位
  韩国非演员、七类情绪、另有 13 人感知评价，官方 Zenodo 元数据为 CC BY 4.0。
  缺点是只给 19 个身体点的全局位置，必须先做 IK/retarget，不能直接监督 18D 关节角。
- **最接近共语目标：USC CreativeIT。** 16 位演员、约 8 个 session，包含双方音视频、
  转录、一方全身 MoCap，以及多人离散和连续维度情绪标注。它规模不大、需要申请且只限
  内部研究，但在“语音/文本/情绪/真实全身动作”四者交集上比其他候选更直接。
- **规模大不代表情绪真值多。** Seamless Interaction 有 4,000+ 小时和 4,000+ 人，
  但当前人工 internal-state 标注只有 1.1 小时第一方和 4.7 小时第三方；其大规模
  `emotion_scores` 是 Imitator 模型输出，不能当监督真值。Embody 3D 有 500 小时真实
  tracked 3D，但必须先统计“动作 + 音频 + 文本 + 情绪”实际交集。
- **很有价值但有现实阻塞：PACO 和 MPI EMBM。** PACO 的“相同动作、不同情绪”
  设计很适合排除动作捷径，但官方数据页没有明确数据许可；MPI EMBM 有自然叙事情绪
  和人类感知标签，但当前 Figshare 下载显示 0 kB。
- **共语风格补充：ZeroEGGS。** 它有同步 WAV + BVH，和本项目的共语手势目标最接近；
  但只有一位演员，标签是混合 style 而非严格情绪真值，许可为 CC BY-NC-ND 4.0。
- **EMOKINE 和 McNorm 只适合单元测试/感知验证。** 两者都是单一舞者，无法支撑
  训练泛化。
- **IEMOCAP 只适合 Qwen/语音/文本情绪表征验证。** 它没有完整上身骨架，不能提供
  18D 动作监督。

因此，若按“今天可研究”排序，第一候选是 Hanyang；若按“最终数据质量”排序，第一
候选是 DIEM-A。两者都不应未经 18D retarget 和机器人可观察情绪复核就混入基础训练。

## 永久数据边界

**Kimodo 不在候选中，也没有被访问、下载或建议。** 后续任何基础训练、微调、评估、
归一化、latent、缓存和 checkpoint 都不能把 Kimodo 当作数据不足时的回退来源。

## Hanyang 人类感知标签实测

为了判断 intended emotion 是否能直接监督生成器，本次只读取官方 743.5 kB 人类评价
表，逐条聚合 4,060 段动作的 13 位观察者选择。结果表明标签噪声不能忽略：

- 单个观察者相对 intended label 的平均正确率：**48.35%**；
- 13 人多数票与 intended label 一致：**56.97%**；
- 至少 70% 观察者同意 intended label：**26.82%（约 1,089 / 4,060）**。

| intended 类别 | 单人平均正确率 | 多数票一致率 | 至少 70% 同意 |
| --- | ---: | ---: | ---: |
| happy | 67.61% | 77.93% | 56.90% |
| sad | 52.58% | 58.79% | 35.86% |
| surprise | 44.59% | 53.45% | 22.07% |
| angry | 52.97% | 61.03% | 31.21% |
| disgust | 35.07% | 45.52% | 8.45% |
| fear | 27.68% | 31.21% | 5.86% |
| neutral | 57.98% | 70.86% | 27.41% |

因此不能把全部 4,060 个 intended label 当作硬标签。安全做法是保留 13 人投票分布作为
soft target，先用人物隔离切分训练独立 critic；若需要高置信监督，只使用约 1,089 个
高一致样本，而且要承认 fear/disgust 几乎没有足够的高置信数据。

## 与文本/语音联合的专项候选

| 数据集 | 规模和关键模态 | 标签风险 / 许可 | 建议用途 |
| --- | --- | --- | --- |
| USC CreativeIT | 16 位演员，约 8 个 session；双方音视频、转录、一方全身 MoCap、多人离散及连续情绪标注 | 需申请；内部研究、非商业、不可转交数据或派生物 | **第一共语候选**：验证文本/Qwen 情绪 latent 是否真正改变同步动作 |
| Embody 3D | 500 小时、439 人、5,400 万帧 tracked 3D；手、体型、分轨音频和文本；含不同情绪状态的对话 | 需填 release form；官方总表没有直接给出四模态情绪交集 | 先只审计 `dataset.json` 元数据，确认交集后再决定是否申请/下载 |
| Seamless Interaction | 4,000+ 小时、4,000+ 人；音频、时间对齐转录、30 Hz SMPL-H | CC BY-NC；3D 来自视频处理；人工标注仅 1.1/4.7 小时，大规模情绪分数为模型输出 | 可做大规模语义/韵律预训练；**模型情绪分数绝不能作为情绪真值** |

## 候选总表

| 排名 | 数据集 | 规模与被试 | 情绪标签 | 动作模态 / 骨架 | 许可与当前获取状态 | 对 18D 上身机器人的判断 |
| --- | --- | --- | --- | --- | --- | --- |
| 1（立即可研究） | Hanyang/Duksung Emotional Body Motion | 4,060 段；29 位韩国非演员；每段 5 秒、30 Hz；另 13 位观察者 | happy、sad、surprise、angry、disgust、fear、neutral | CSV；19 个身体部位的全局 XYZ 位置 | 官方 Zenodo 为 Open；REST 元数据标 `cc-by-4.0`；304.8 MB 动作 + 743.5 kB 人评文件，当前下载项存在 | **中高**：类别和人物较均衡，人评很有价值；但无局部旋转，必须经 IK、轴向/尺度标定和机器人 QC。优先用作 emotion critic/校准，不直接作 18D 角度真值 |
| 2（质量第一，需申请） | DIEM-A | 官方项目页 10,767 段；当前分发卡 10,212 段；97 位专业演员，54 日本、43 台湾 | 12 类 + neutral；每类 3 个个性化场景，低/中/高三档强度；日/中/英文本 | Vicon 57 markers；重建 24 joints；BVH/C3D/FBX | 签署 ULA 后申请；签署人须为学术/公立研究机构 PI；只限非商业学术研究；禁止再分发；当前分发约 40.9 GB | **很高**：规模、亚洲人群、强度和文本最符合目标；需 24→18D retarget、逐关节 Euler order 处理和许可隔离 |
| 3（许可待确认） | PACO Motion Library | 原始论文报告 4,080 个 movement；30 位非专业演员，15 女 | angry、happy、sad、neutral | 60 Hz CSM；另有 15 joints × XYZ 的 PTD；walk/knock/lift/throw/sequence | Glasgow 官方页仍有 1.02 GB CSM 和 PTD 下载入口；页面未给出明确数据许可证 | **中高**：同一动作跨情绪的实验设计很适合检验 emotion/action 解耦；PTD 只有位置，需要 IK。书面许可前不下载、不训练 |
| 4（下载故障） | MPI Emotional Body Expressions Database (MPI EMBM) | 最终 1,447 段、约 86 分钟；8 位业余演员（4 女） | amusement、joy、pride、relief、surprise、anger、disgust、fear、sadness、shame、neutral；含 intended 与 perceived 标签 | Xsens MVN，120 Hz；23 body joints；BVH/MVNX；位置、方向及动力学量 | 论文称数据开放；Figshare 标 CC BY 4.0，但当前页面显示 `Download (0 kB)`，旧官方站点也未确认可用 | **高（若恢复）**：叙事/独白式上身动作和人类感知标签很贴近目标；被试较少。先联系作者/Figshare 恢复官方文件 |
| 5（受限研究） | PhysioNet Kinematic Actors | 1,402 段；22 位半专业演员（11 女）；每段约 6 秒，125 Hz | happy、sad、angry、fear、disgust、surprise、neutral；自由表演 + 5 个场景 | Noitom Perception Neuron 17 sensors；BVH 72 nodes（root、58 joints、13 end sites），含位置和旋转 | Restricted Access；注册用户须签 Restricted Health Data License/DUA 1.5.0 | **高（技术）/中（目标）**：骨架可直接 retarget，但片段短、非共语；只能在独立合规研究环境使用 |
| 6（共语风格补充） | ZeroEGGS / ZEGGS | 67 个英语独白序列；134.65 分钟；1 位女性演员 | 19 种 style；含 angry、happy、sad、scared、neutral，也混有 sarcastic、old、tired 等 | 同步 WAV + 全身/手指 BVH；官方仓库提供 raw 和 clean | 官方 Git LFS/分卷 ZIP 当前存在；CC BY-NC-ND 4.0，只限非商业且禁止分享 adapted material | **动作高/情绪真值低**：共语最匹配，但单人和混合 style 会把身份/表演习惯当成情绪。只适合作风格/语音条件研究，不能作为主情绪监督 |
| 7（验证集） | EMOKINE Pilot | 63 段：9 个 choreography × 6 情绪 + 9 个说明；1 位专业女性舞者 | anger、contentment、fear、joy、neutral、sad；带观察者验证 | Xsens Link 17 sensors；23 keypoints，240 Hz；MVNX、CSV、渲染刺激和运动学特征 | 官方 Zenodo 为 Open；REST 元数据标 `cc-by-4.0`；1.7 GB 文件当前存在 | **中（验证）**：7/9 choreography 主要是手臂，同动作跨情绪非常适合 sanity check；单人、63 段，不能承担训练 |
| 8（验证集） | McNorm | 原始 85 段（17 dance × 5 情绪）；剔除缺失/采集问题后 73 段；1 位专业女性芭蕾舞者 | angry、fear、happy、sad、neutral；带观察者评价 | Vicon 12 cameras、39 markers、120 Hz；论文分析使用简化 15-marker 轨迹 | 官方 OSF 项目公开，但当前机构元数据未给出清晰数据许可证 | **低（训练）/中（测试）**：只适合小型情绪识别 benchmark；舞蹈域和单人偏差太强，书面许可前不接入 |
| 9（只用于 Qwen） | IEMOCAP | 约 12 小时；10 位演员、5 个双人 session | anger、happiness、excitement、sadness、frustration、fear、surprise、other、neutral；另有 valence/arousal/dominance，多人标注 | 音频、视频、转写；mocap 仅一方的 face、head、hand，不是完整身体 | 填申请表并签 USC 协议；内部研究、非商业；未经许可不得转交数据或派生物 | **低（18D）/高（语义）**：缺 shoulder/elbow/torso 真值，不能做动作监督；可单独验证 Qwen 文本/语音情绪 latent |

## 排序依据

这里不是简单按“文件最多”排序，而是同时看五个维度：

1. **情绪真值可靠性**：是否有人类感知评价、强度标签，以及 intended/perceived
   emotion 是否分开。
2. **人物泛化**：必须按 actor/source group 划分；单演员数据不能证明新人物泛化。
3. **动作可迁移性**：是否覆盖 pelvis、躯干、头、肩、肘、腕，是否有旋转而非仅全局
   位置。
4. **目标域匹配**：共语/叙事上身动作优于舞蹈和孤立模板动作。
5. **许可与今天的可获得性**：官方页面存在不等于允许训练和发布权重；下载失效也不能
   按“公开”处理。

### 数据质量排序

1. DIEM-A：人物、类别、强度、文本和骨架最完整。
2. CreativeIT：四模态交集最直接，但样本较小、许可严格。
3. Hanyang Emotional Body Motion：开放、均衡、有人评，但实测标签噪声高且需要 IK。
4. MPI EMBM：自然叙事和 perceived emotion 很强，受当前下载故障与 8 人规模限制。
5. Embody 3D：真实 3D 规模最大，但四模态情绪交集尚未核实。
6. PACO：动作控制实验最干净，但许可不明。
7. PhysioNet Kinematic Actors：技术格式好，许可受限且片段短。
8. Seamless Interaction：共语规模大，但不是直接情绪真值。
9. ZeroEGGS：共语价值高，但不是可靠的多人物情绪真值。
10. EMOKINE / McNorm：只作验证。
11. IEMOCAP：只作 Qwen/语音/文本表征验证。

### 当前可执行性排序

1. Hanyang：官方文件存在、开放、CC BY 4.0；但仍需保存许可快照和完成 19 点→18D
   预研后再决定是否下载。
2. EMOKINE：同样开放且文件存在；只建议用来做 retarget/critic 单元测试。
3. DIEM-A：先确认项目是否满足非商业学术条件，再由 PI 申请。
4. MPI EMBM：先解决官方数据下载故障。
5. PACO：先取得书面数据授权。
6. PhysioNet/IEMOCAP/ZeroEGGS：仅限明确许可边界内的隔离研究。

## 为什么不能直接把外部数据混入基础训练

这些数据集的条件和观测空间不同：

- 情绪标签包含“演员意图”“观察者感知”“风格”“角色状态”等不同概念；
- BVH rotation、marker XYZ、segment orientation 和 19-joint global XYZ 不是同一种
  监督；
- 30/60/120/125/240 Hz、坐标轴、Euler 顺序和片段时长差异很大；
- human full-body emotion 不一定在机器人受限 18D 上仍可观察；
- dance、walk、lift 等动作模板可能成为捷径，导致分类指标高而真正情绪控制失败。

更安全的路线是：

1. 先训练或验证一个与生成器隔离的 **emotion critic / representation adapter**；
2. 每个数据集单独 importer、normalizer、actor-group split 和 18D retarget 目录；
3. 对 retarget 后的机器人动作做盲评，只给人类能够从机器人动作看出的样本开启情绪
   supervision mask；
4. 先做 `BEAT2-only`、`external critic only`、`external adapter` 三组 A/B，不共享
   外部数据 normalizer；
5. 必须报告 correct-vs-shuffled emotion、按 actor 泛化、动作类别捷径、幅度、速度、
   jerk、关节限位和 60 秒漂移；不能只报训练 loss。

## 官方来源

### DIEM-A

- [Tohoku/RIEC 官方项目页](https://www.cr-ict.riec.tohoku.ac.jp/diem-a/)
- [RIEC 官方分发卡、格式和 ULA 条件](https://huggingface.co/datasets/RIEC/DIEM-A)
- [ACII 2025 原始论文 PDF](https://www.riec.tohoku.ac.jp/~kitamura/PDF/DIEM-A_ACII2025_Cheng.pdf)

### Hanyang/Duksung Emotional Body Motion

- [官方 Zenodo 数据记录](https://zenodo.org/records/10052504)
- [官方 Zenodo REST 元数据（许可和文件状态）](https://zenodo.org/api/records/10052504)
- [原始论文 DOI](https://doi.org/10.1109/TAFFC.2024.3365895)
- [Hanyang 机构论文记录](https://scholarworks.bwise.kr/hanyang/handle/2021.sw.hanyang/212031)

### PACO

- [University of Glasgow PACO Lab 官方数据页](https://paco.psy.gla.ac.uk/?page_id=14973)
- [原始数据论文的 UEA 机构记录](https://ueaeprints.uea.ac.uk/id/eprint/90416/)
- [原始论文 DOI](https://doi.org/10.3758/BF03192758)

### MPI Emotional Body Expressions

- [PLOS ONE 原始论文和数据说明](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0113647)
- [作者 Figshare 数据记录](https://figshare.com/articles/dataset/MPI_EMBM_Database_Mocap_Files/1220428)

### PhysioNet Kinematic Actors

- [PhysioNet 官方 v2.1.0 数据页](https://www.physionet.org/content/kinematic-actors-emotions/2.1.0/)
- [Scientific Data 原始论文](https://www.nature.com/articles/s41597-020-00635-7)

### ZeroEGGS / ZEGGS

- [Ubisoft La Forge 官方仓库和数据说明](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS)
- [官方许可证](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS/blob/main/License.md)
- [Ubisoft 官方项目介绍](https://www.ubisoft.com/en-us/studio/laforge/news/5ADkkY0BMG9vNSDuUMtkeg/zeroeggs-zeroshot-examplebased-gesture-generation-from-speech)

### EMOKINE

- [官方 Zenodo 数据记录](https://zenodo.org/records/7821844)
- [官方 Zenodo REST 元数据（许可和文件状态）](https://zenodo.org/api/records/7821844)
- [作者官方代码仓库](https://github.com/andres-fr/emokine)
- [Behavior Research Methods 原始论文](https://link.springer.com/article/10.3758/s13428-024-02433-0)

### McNorm

- [作者 OSF 数据项目](https://osf.io/458sq/)
- [原始论文](https://link.springer.com/article/10.1007/s00426-022-01669-9)

### IEMOCAP

- [USC SAIL 官方信息页](https://sail.usc.edu/iemocap/iemocap_info.htm)
- [USC SAIL 官方发布和申请页](https://sail.usc.edu/iemocap/iemocap_release.htm)
- [USC 原始许可协议](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf)

### USC CreativeIT

- [USC SAIL 官方发布页](https://sail.usc.edu/CreativeIT/ImprovRelease.htm)
- [USC CreativeIT 数据许可](https://sail.usc.edu/CreativeIT/Data_Release_Form_CreativeIT.pdf)

### Embody 3D

- [Meta 官方数据工具仓库与数据说明](https://github.com/facebookresearch/embody-3d)

### Seamless Interaction

- [Meta 官方数据仓库、模态、人工标注规模和许可](https://github.com/facebookresearch/seamless_interaction)

## 本次未执行的动作

- 未下载任何动作数据包；只在系统临时目录检查了 Hanyang 的小型人评表；
- 未提交 DIEM-A、PhysioNet 或 IEMOCAP 申请；
- 未启动、停止或修改训练；
- 未创建外部数据 normalizer、split、cache 或 checkpoint；
- 未访问或使用 Kimodo。
