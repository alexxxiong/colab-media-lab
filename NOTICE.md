# 来源、署名与使用范围

本仓库公开研究报告、测量数据、提示词及脱敏后的部署参考，不发布模型权重、私人环境归档或可转用的账号授权。

## FastVideo 与模型

- FastVideo：<https://github.com/hao-ai-lab/FastVideo>，固定提交 `ce970eb715215c79a169658d7aa6c8a96e64891e`。参考实现或构建定义涉及其代码；上游 Apache-2.0 许可文本保存在 [licenses/FastVideo-APACHE-2.0.txt](licenses/FastVideo-APACHE-2.0.txt)。
- 模型：<https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer>。模型页面标记为独立许可；本仓库不重新授权模型或权重，也不证明读者满足其使用条件。
- Colab CLI：<https://github.com/googlecolab/google-colab-cli>。这是外部安装依赖，非本仓库自研服务。

没有给整个仓库或所有第三方材料统一套用 MIT 许可。对新编写脚本和报告的额外许可授权可由仓库维护者另行声明；第三方权利和已有许可保持原状。

## 鲸鱼娘

作者：**YunYueSama**  
仓库：<https://github.com/YunYueSama/codex-deepseek-pet>  
参考 revision：`7661c8b304c5400701f91da01b1a643a207331de`

本实验依据社区设定生成新首帧，并添加饭碗构图和《先吃饭，再思考》动作剧本。视频和首帧为经过修改的 AI 生成衍生内容，不能冒称上游作者作品或 DeepSeek 官方设定，也不暗示上游作者或 DeepSeek 官方背书。

署名许可覆盖上游有权授权的新增贡献，不增加底层第三方参考作品、标志等权利；完整许可与素材范围说明保存在 `licenses/community-character-*`。公开包未附上游的原始社区参考图。

## 结论范围

所有技术与成本结论只针对本次固定版本、输入与已测规格。三条主片没有通过完整内容验收；正式矩阵、原精度模型、微调、Colab 生图及大规模产能均不属于已验证结果。
