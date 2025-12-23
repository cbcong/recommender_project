# recommender_project

## 改进策略与动机（准确性 + 多样性）

### 研究级思路速览
- **多目标训练（Multi-Objective）**：在主准确度损失（如 MSE/BPR）之外，引入覆盖度、ILAD 或类别分布的辅助项，用可调或自适应权重（可结合 MGDA）压制同质化推荐，兼顾精度与新颖度。
- **知识蒸馏 + 重排序**：以 MMR/DPP 等确定性多样性算法作为教师信号，学生模型学习多样性概率分布；推理端线性融合原始相关性与多样性分数，无需昂贵的贪心重排。
- **对比学习与邻域相似性**：在内容/协同对齐的基础上，引入用户或物品层面的对比任务，提升长尾物品和多兴趣用户的可分性（可借鉴 GraphDR、HybridAcc 思路）。
- **可学习 + 自适应流行度惩罚**：HybridTail 允许 α 可训练、用户依赖、并提供动量和平滑上界，抑制热门偏置的同时避免对整体准确率造成二次伤害。
- **注意力/门控融合**：在 HybridNCF 的协同-内容融合处加入注意力或门控，为不同用户自适应分配两类特征权重，提升个性化表达。
- **多通道候选与聚合**：用多子模型或多次随机采样生成差异化候选集，再用 MMR/投票聚合，提高覆盖度与多样性。

### 关键实现要点（本仓库）
- **HybridTail 自适应流行度抑制**：`learnable_pop_alpha` 允许 α 参与训练；`user_pop_scaling` + `user_pop_pref_momentum` 基于用户历史平均流行度做动态缩放并支持动量平滑；`pop_penalty_cap` 可对惩罚设置上界避免梯度爆炸。
- **配置即插即用**：在 `utils/config.yaml` 的 `hybrid` / `hybrid_tail` 节下可直接配置上述开关与范围；`experiments/ml1m/train_hybrid_tail.py` 自动读取并写入 checkpoint，便于评估端复现。
- **训练脚本兼容 AMP 与历史缓存**：训练使用与推理一致的打分逻辑，结合用户历史缓存降低评估开销，同时保持梯度一致性。

### 快速运行示例
```bash
# 训练 HybridTail（默认启用用户自适应流行度缩放，可选动量/上界）
python experiments/ml1m/train_hybrid_tail.py
```
可在 `utils/config.yaml` 中调整：
- `hybrid_tail.learnable_pop_alpha`: 是否学习 α；
- `hybrid_tail.user_pop_scaling`: 是否按用户历史做缩放；
- `hybrid_tail.user_pop_pref_momentum`: 用户流行度偏好动量（0 表示直接替换，>0 表示平滑更新）；
- `hybrid_tail.pop_penalty_cap`: 惩罚上界（空值表示不封顶）。
