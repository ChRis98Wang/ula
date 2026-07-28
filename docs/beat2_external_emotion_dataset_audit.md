# BEAT2 外部情绪动作数据集只读接入审计

审计日期：2026-07-27
范围：MPI Emotional Body Expressions、DIEM-A、ZeroEGGS/ZEGGS、XEM，以及
PhysioNet Kinematic Actors、IEMOCAP、PACO Body Movement Library。
操作边界：本次只查官方项目页、作者仓库、原始论文或机构数据仓库；没有下载数据包，
没有修改训练器，没有把外部数据、缓存或统计量写入生产数据。

## 结论

当前没有任何外部数据集获准直接进入生产训练：

- **最值得申请和验证：DIEM-A。** 规模和标签设计最好，具有 12 种情绪、中性、
  低/中/高强度以及三语场景文本；24 关节 BVH 很适合转 18D。但其 ULA 限定学术
  非商业、PI 签署且禁止再分发，因此只能在获批后的隔离研究环境使用，不能默认进入
  商业或可发布权重。
- **最适合开放许可技术试接：MPI EBEDB、XEM。** 两者官方数据页分别标为
  CC BY 4.0 和 Etalab Open License 2.0。MPI 更自然、情绪更丰富且带文本和人类
  感知标签；XEM 下载最直接，但规模小、动作与情绪严重耦合，只适合作为校准集。
- **ZeroEGGS 不是理想的情绪真值集。** 它的优势是 134.65 分钟英文语音和风格化
  共语手势；只有一位演员，19 个标签混合了情绪、态度、角色和状态，也没有强度真值。
  CC BY-NC-ND 4.0 进一步限制了生产使用和派生物发布。
- **PhysioNet Kinematic Actors** 有良好的七类全身动作，但受限于只允许科学研究、
  不得共享数据的协议。
- **IEMOCAP** 适合文本/语音情绪表征验证，不适合作为 18D 身体动作真值：其 mocap
  只覆盖面部、头部和手部，缺少完整肩肘腕/躯干链。
- **PACO** 的动作数据与 18D 有一定兼容性，但官方数据页未给出明确的数据许可，
  在获得书面授权之前保持禁用。

近期应继续以 **BEAT2-only** 为训练主线，并先完成人工机器人可观察情绪复核。外部
数据不能解决明早结果，也不应在未完成许可和质量闸门时临时混入。

## 永久拒绝项

**Kimodo 永久排除。** 新的基础训练、后训练、微调、评估和归一化统计均不得读取
Kimodo 的原始文件、派生轨迹、缓存、latent、split、checkpoint 或由其训练出的权重。
即使 BEAT2 或外部情绪数据数量不足，也不得把 Kimodo 作为回退来源。

## 快速比较

| 数据集 | 官方规模 | 动作与文件 | 文本/语音/强度 | 许可与获取 | 18D 适配判断 | 当前决定 |
| --- | --- | --- | --- | --- | --- | --- |
| MPI Emotional Body Expressions (EBEDB) | 最终库 1,447 段；8 位演员；120 Hz | 全身 23 关节；BVH，旧交互库另有 MVNX | 有动作动机/叙事片段文本；无公开音频；无离散强度，带 intended/perceived emotion、观察者响应和一致率 | Figshare 标为 CC BY 4.0；官方旧数据库地址可用性需重新确认 | **高**：头、躯干、肩、肘、腕链齐全；需 120→30 Hz、轴/骨架标定和 GMR/物理 QC | 许可上可进入试接候选；数据可获得性和条款快照通过前不接入 |
| DIEM-A | 官方项目页称 10,767 段；当前分发卡列 10,212 段、40.9 GB；97 位专业演员 | 24 关节 BVH；57 标记 C3D；骨架+标记 FBX | 12 情绪＋中性；每情绪 3 场景×低/中/高；日/中/英场景文本；分发卡不含视频 | 签署 ULA 后申请；仅学术非商业；PI 必须签署；禁止再分发 | **很高**：骨架和标签最完整；需处理逐关节 Euler 顺序和少量不同采集系统样本 | 仅获批后的隔离研究；生产/商业训练禁用 |
| ZeroEGGS 项目的 ZEGGS | 67 个独白序列，134.65 分钟；1 位英语女性演员 | BVH 动画＋WAV 语音，提供 raw/clean 对齐版本 | 19 个 style；含 angry/happy/sad/scared，也含 sarcastic/old/tired 等；无强度；官方仓库未声明发布转写文本 | 仓库直接用 Git LFS/分卷 ZIP；CC BY-NC-ND 4.0 | **高（动作）/低（情绪真值）**：共语上半身很匹配，但单演员和混合 style 标签导致强偏差 | 只作非商业风格研究候选；不得当作正式情绪监督或进入生产权重 |
| XEM | 10 人×5 动作×4 情绪×5 重复＝1,000 段；259.2 MB | XSens 23 段；MAT 中含 3D 坐标、线/角速度、Euler、质心 | angry/neutral/happy/sad；无语音、文本或强度；五种固定动作 | 官方仓库直接下载；Etalab Open License 2.0（兼容 CC BY 2.0） | **中高**：关节覆盖足够，但需 MAT importer；动作类别与情绪完全交叉但仍可能产生动作捷径 | 可做最小开放许可校准试验；不得单独证明情绪泛化 |
| PhysioNet Kinematic Actors | 1,402 段；22 位演员；每次表演约 6 秒；125 Hz | Perception Neuron；BVH，72 anatomical nodes | happy/sad/angry/fear/disgust/surprise/neutral；自由＋五场景；无语音、文本或强度 | 注册并签 Restricted Health Data License/DUA；仅合法科学研究，不得共享访问 | **高**：完整头、躯干、双臂；需 125→30 Hz 和 72 节点映射 | 仅隔离科学研究，不能默认进入生产或共享权重 |
| IEMOCAP | 约 12 小时；10 位演员；即兴和脚本双人对话 | 双路音视频；mocap 仅一方的面部、头部和手部 | 完整转写及词/音素/音节对齐；多标注者类别和维度情绪 | 申请并签 USC 协议；仅内部研究、非商业，数据及派生物不得转交 | **低（18D）/高（Qwen/语音验证）**：缺完整肩肘腕和躯干链 | 不用于动作监督；可在独立许可环境评估文本/语音情绪 latent |
| PACO Body Movement Library | 官方页提供 1.02 GB CSM；演员总数/总段数未在数据页明确汇总 | 60 Hz CSM；另有 15 关节 3D 位置 PTD；lift/knock/throw/sequence/walk | angry/happy/sad/neutral；无文本、语音或强度 | 官方实验室直接下载，但数据页未声明明确数据许可 | **中**：位置轨迹可间接 IK，CSM 需专用解析；固定动作混淆明显 | 许可不明，保持禁用并请求书面授权 |

## 官方依据与逐项说明

### 1. MPI Emotional Body Expressions Database

MPI/作者论文记录说明该库包含自然独白式情绪身体动作，120 Hz、23 个身体关节的
逐帧位置与方向；最终可筛选的库有 1,447 段，包含 intended emotion、观察者
perceived emotion、完整响应分布、动作物理属性和 acting motivation/text。
录音时演员会讲述故事，但因隐私只公开 mocap 和文本标注，不应假设存在可训练音频。

- [Max Planck 原始出版记录](https://pure.mpg.de/pubman/item/item_2160764)
- [PLOS 原始论文及数据说明](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0113647)
- [作者 Figshare 数据记录：BVH、CC BY 4.0](https://figshare.com/articles/dataset/MPI_EMBM_Database_Mocap_Files/1220428)

风险：Figshare 页面目前显示归档说明但下载大小为 0 kB，旧
`ebmdb.tuebingen.mpg.de` 入口也需要重新验证。因此“论文称公开”和“今天可稳定批量
取得”不能视为同一件事。接入前必须先向作者/Max Planck 确认当前官方下载位置和
元数据许可范围。

### 2. DIEM-A

这是标签设计上最符合“风格＋情绪＋强度＋文本”的候选。官方项目页给出 97 位日本/
台湾专业表演者、12 情绪和中性、每情绪三个自拟场景及三档强度、日中英场景文本。
项目页的总量是 10,767；机构官方分发卡当前列 10,212 条，并说明部分不同设备采集者
稍后发布，这很可能解释差额，但接入时仍必须以获批下载后的 manifest 为准。

分发格式为 24 关节 BVH、57 标记 C3D、骨架和标记 FBX。2026-06 后 BVH/FBX 改为
逐关节 Euler rotation order，不能沿用论文里统一 ZYX 的假设。

- [Tohoku/RIEC 官方项目页](https://www.cr-ict.riec.tohoku.ac.jp/diem-a/)
- [RIEC 官方申请与数据卡](https://huggingface.co/datasets/RIEC/DIEM-A)

许可明确限制为学术非商业研究、教学和课堂使用；要求学术/公立研究机构 PI 签署，
禁止再分发。未取得书面批准前不能下载；取得后也必须存放在与生产数据、发布权重
完全隔离的研究空间。

### 3. ZeroEGGS / ZEGGS

ZeroEGGS 是方法名，随仓库发布的数据集名为 ZEGGS。官方仓库给出 67 个英语独白、
一位女性演员、19 种 motion style、合计 134.65 分钟；原始和清理后的 BVH 与 WAV
都在 Git LFS 分卷中。它的真正价值是语音—手势—风格联合学习。

其标签不是纯情绪本体：`angry/happy/sad/scared` 与
`agreement/disagreement/sarcastic/threatening/old/tired/still` 等混在同一 style
空间。没有低/中/高强度真值，单演员也无法分离人物风格和情绪风格。因此不能把它
当成扩大六情绪监督数量的直接答案。

- [Ubisoft La Forge 官方仓库和数据说明](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS)
- [仓库原始许可证：CC BY-NC-ND 4.0](https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-ZeroEGGS/main/License.md)

许可证只允许非商业使用；允许非商业地制作但不公开分享 adapted material。训练权重
是否构成/包含不可分享的派生物需要法律审查，默认不得进入生产或可发布 checkpoint。

### 4. XEM

XEM 的实验设计是 10 位参与者分别以 angry、neutral、happy、sad 表演 dancing、
hands-up、waving、stopping、pointing，每种组合重复五次，共 1,000 段。XSens
23-segment 数据保存在一个 MAT 文件，含 3D 位置、线速度、角速度、Euler 角和质心。

- [作者维护的 XEM 官方说明和数据结构](https://github.com/evraplatform/XEM-dataset/wiki)
- [法国国家研究数据仓库记录、下载和许可证](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FGZQCOY)

开放许可清晰、体积小，适合先写一个完全隔离的 importer/retarget smoke test。主要
问题是动作模板过少；若采样或 split 不严，模型可能用动作/参与者识别标签，而不是
学习机器人可观察的情绪动力学。

### 5. PhysioNet Kinematic Dataset of Actors Expressing Emotions

官方 v2.1.0 页面给出 22 位演员、1,402 条、125 Hz、每次约六秒，七类情绪，BVH
包含位置和旋转。全身链足够转成 pelvis、双肩、双肘、双腕和 head 的 18D。

- [PhysioNet 官方数据页](https://physionet.org/content/kinematic-actors-emotions/2.1.0/)
- [Restricted Health Data License 1.5.0](https://www.physionet.org/content/kinematic-actors-emotions/view-license/2.1.0/)

许可只允许科学研究，禁止共享访问，并要求维护数据安全和公开相关论文代码。它可以
是学术复现实验，但不是生产训练的无障碍来源。

### 6. IEMOCAP

IEMOCAP 的优势是约 12 小时情绪对话、音视频、完整转写、细粒度对齐以及多标注者的
类别/维度情绪标签。但 USC 官方页明确说 mocap 只覆盖面部、头和手，并非完整身体。
它无法提供可靠的 pelvis/shoulder/elbow/wrist 关节角真值。

- [USC SAIL 官方发布页](https://sail.usc.edu/iemocap/iemocap_release.htm)
- [USC 原始许可协议](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf)

协议限制内部研究、禁止商业使用，且未经许可不能转交数据或派生物。若未来验证 Qwen
情绪 latent，可在独立研究实验中用其文本/语音标签做分类或检索评估，但不可把缺失的
全身动作伪造成 18D 监督。

### 7. PACO Body Movement Library

PACO 官方实验室提供 60 Hz CSM 和 15 关节位置 PTD，含五种动作与 angry/happy/
sad/neutral。它有明确的动作起止帧，适合研究“相同动作、不同情绪”，但官方数据页
没有声明数据许可，网站通用条款不能替代数据授权。

- [University of Glasgow PACO Lab 官方数据页](https://paco.psy.gla.ac.uk/?page_id=14973)

在作者书面确认训练、派生轨迹和权重发布权限之前，不下载、不导入。

## 若后续获批，必须经过的接入闸门

1. **许可闸门**：保存官方许可、申请批准、版本、下载 manifest 和 SHA256；明确研究/
   商业、派生物、权重发布及再分发权限。许可不明即关闭。
2. **物理隔离**：每个外部集使用独立 raw/processed/cache/run 根目录和 dataset ID；
   不得覆盖 BEAT2 split、normalizer 或 checkpoint。
3. **骨架闸门**：显式记录原坐标系、Euler order、帧率和骨架；只从原始完整段
   retarget 到当前 18D，禁止复制窗口凑数。
4. **质量闸门**：运行现有 joint limit、velocity、target fit、collision、axis/head
   direction、continuity 和 endpoint 检查；安全 retime 必须有完整审计。
5. **标签闸门**：官方/文件名 emotion 只作为候选元数据。只有盲测下两位独立审核员
   对机器人动作的可观察情绪及类别一致，或由独立第三人裁决后，监督 mask 才可开启。
6. **无泄漏 split**：按演员/原始录制 source group 固定划分；同一演员、同一长录制
   或其任何派生片段不得跨 train/validation/test。
7. **A/B 闸门**：先做 BEAT2-only 基线与“单一外部集 adapter”对照，检查
   correct-vs-shuffled condition、类别混淆、速度/幅度、jerk、动作模板捷径和身份捷径。
   未通过不得混入基础网络。

## 推荐次序

1. 继续完成现有 BEAT2 机器人可观察情绪人工复核，不影响当前训练。
2. 联系 MPI 作者确认当前可用的官方 BVH/metadata 下载及 CC BY 4.0 范围。
3. 用 XEM 做不超过一个小批次的离线 importer/retarget 合约验证；不训练生产模型。
4. 若项目确属合规的非商业学术研究，由 PI 申请 DIEM-A；否则停止。
5. ZeroEGGS、PhysioNet、IEMOCAP 只在许可隔离的研究分支考虑，PACO 等待书面许可。
