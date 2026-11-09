# 部署参考与已遇到的问题

这些脚本来自实际研究流程，公开副本只做脱敏、显式 GPU 开关和保守示例预算的修改。**没有为发布重新启动 GPU，也不保证脱离本次固定版本后一键运行。** `data/` 是历史测量，`projects/` 是可审查的部署参考，二者不要混淆。

## 工具与目录

本次使用官方 [google-colab-cli](https://github.com/googlecolab/google-colab-cli)，实验固定版本 `0.7.5`。上游当前说明 Linux/macOS 受支持；实验在 Windows 的 WSL2 / Ubuntu-24.04 中运行 CLI。浏览器用于用户自身登录及 Drive 授权，凭据只保留在本机受限配置目录。

```text
Colab-Lab/                    # 自己的 Google Drive 目录
  datasets/                   # 冻结首帧、尺寸派生、剧本和输入清单
  models/base/                # 基础模型，按 revision 保存
  models/adapters/            # 如以后训练 LoRA，单独保存适配器
  models/merged/              # 仅确有独立部署需求时保存合并模型
  notebooks/minimax-h3/cache/  # 自己构建并校验的环境和内核存档
  runs/                       # 请求、媒体、资源数据和校验结果
```

`/content/colab-lab` 是临时盘。先把 Drive 上的所需权重暂存并校验到临时盘，再推理；完成后发布到 Drive 并核验，再释放。Drive 扩容不等于 CU 增加。

## CPU 可先做的准备

1. 在本机完成账号登录、优惠资格确认、许可核对、输入剧本和依赖版本整理。
2. 按固定模型 revision 准备权重清单；数据和权重放自己的 Drive，不提交 Git。
3. 在 CPU 准备环境完成可行的下载、Python 环境构建和权重视图整理；临时目录不能跨 Colab VM 直接继承。
4. 将自己的环境打成归档，保存 SHA-256、字节数、源代码提交和依赖记录。GPU 节点恢复时先校验，再用原生 `tar` 解包。

参考 `projects/minimax-h3/prepare_environment_cpu.sh`。它是历史准备脚本，不是完整的最终依赖锁恢复器；最后环境的观测版本在 `deployment/environment-observed.json`。依赖元数据未捕获的项目明确保留为未捕获，不补造版本号。

## 最终运行路线

- Python 3.12 与模型 Torch 环境独立于 CLI 环境；观测 Torch 为 `2.12.0+cu130`。
- FastVideo 提交和模型 revision 都固定，禁止仅根据浮动 `main` 推断本实验可复现。
- 主体采用 NVFP4 与 H3 稀疏注意力。`kernel_overlay.py` 在导入 FastVideo 前检查匹配的原生 ABI 和 Python API。
- 模型保持原始文件；`prepare_adaln_view_cpu.py` 另外构建校验过的活动视图，不能直接删除原 checkpoint 内的投影后任意换时间表。
- `adaln_i2v.py` 补齐图生参考时钟的原生固定表。构建需要原始 BF16 投影和足够的主机内存；正式采样只加载已完成校验的表，禁用构建器。
- 每条请求重新处理首帧与文本，关闭输出缓存和跨请求条件特征复用，保留模型及内核缓存。

官方首帧接口与本次请求形态：

```python
InputConfig(pil_image=image, last_image=None)
# SamplingConfig: fps=24, num_frames=124/243, num_inference_steps=9,
# guidance_scale=1.0, batch_cfg=False; 9 grid points -> 8 DiT forwards.
```

## 三个实际阻塞点

1. **Ninja PATH**：环境中已有构建工具，但工作进程 PATH 未包含受管 venv 的 `bin`。补齐 PATH 后复测。
2. **T2V 表不覆盖 I2V 时钟**：原表对当前首帧参考时钟不适用，不能插值或简单关闭表，因为活动权重视图已不含大投影。使用原始投影的原生计算补齐并校验。
3. **FP4 wheel 与源码 API 不匹配**：已安装 wheel 缺少固定源码使用的 Python 接口。`build_fp4_kernels_remote.py` 从固定 FastVideo 与 CUTLASS 子模块提交构建两个所需扩展及匹配 API，缓存校验过的覆盖包；不整体升级依赖来碰运气。

## 会话和成本控制

`budget_guard.py` 与 `cpu_prepare_session.py` 通过固定的本次 notebook UUID 识别自己创建的资源，独立看门狗监控费率、余额与停滞，并核验释放。不能终止其他实验的会话。

H3 控制器指定 G4，不自动换卡；通用 guard 中保留的 T4 smoke 是早期合成验证入口，不是 H3 测试。公开副本默认拒绝 GPU 分配，且示例 H3 预算设为 900 秒 / 2.5 CU。它不适合直接执行完整矩阵，读者需先规划自己的实验。历史图生研究曾明确取消单轮上限，这一历史事实单独记录，不能套用到公开默认值。

```bash
# 本地安装 CLI（不分配 VM）
python -m venv .venv
. .venv/bin/activate
pip install 'google-colab-cli==0.7.5'
colab --help
colab usage
```

不要把 `colab new`、`colab run` 或控制器 `start` 写进常规文档检查和 CI。真正申请硬件前，必须配置自己的许可状态、输入、缓存、运行目录和预算；本仓库不提供可继承的账号授权或私人环境归档。公开副本的 `COLAB_MEDIA_ALLOW_GPU` 只用于明确区分“审查代码”和“主动申请资源”，本次发布没有开启它。

费用限制是本地提前停止目标，不是 Google 的账单硬封顶。看门狗依赖本机/WSL 继续运行与网络可用，关闭电脑或反复网络故障可能影响它。用完还要检查当前账户的会话与费率。

## 捕获与退出

`supervisor.py` 分别捕获 stdout、stderr、真实子进程 PID 和退出状态。逐请求 JSON、完整解码和 SHA-256 决定媒体技术结果。正常停止时先等待当前推理进程退出，再取回证据、执行官方 `drive.flush_and_unmount`、释放本次 VM，并由独立看门狗复核。

遇到 CLI 内核消息等待挂起时，研究中通过官方 CLI `console` 执行同一 Drive 刷写函数。长会话只刷新自己已有实例的代理连接；不能以刷新凭据为名重分配 VM。TLS 读取故障采用有界重试，保留证书验证，不自动重发推理任务。

## 公开副本与历史环境的差异

私人账号、Drive ID、授权确认、机器会话标识、绝对本机路径和原始日志未随仓库发布。示例 `project-config.example.json` 的授权状态默认未配置；`session-options.example.json` 需要自己的代码包摘要。原始私人环境归档没有发布，`bootstrap_remote.py` 因此改为验证读者自己生成的归档摘要。公开副本的源文件哈希不等于原始执行包哈希，不能声称它已在云 GPU 上重新验收。
