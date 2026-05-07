import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 模块一：标准化张量预处理流 (Dataset)
# ==========================================
class GridTopologyDataset(Dataset):
    def __init__(self, jsonl_path, max_nodes_pad=32):
        self.data = []
        self.max_nodes_pad = max_nodes_pad
        
        # 将数据加载到内存
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        user_content_str = sample["messages"][1]["content"]
        grid_data = json.loads(user_content_str)
        
        # 提取目标矩阵和掩码
        target_matrix = torch.tensor(grid_data["target_adj_matrix"], dtype=torch.float32)
        node_mask = torch.tensor(grid_data["node_mask"], dtype=torch.float32)
        
        # 提取特征：为每个节点构造 [P_mw, is_feeder] 
        node_features = torch.zeros((self.max_nodes_pad, 2), dtype=torch.float32)
        for node in grid_data["nodes"]:
            n_idx = node["node_id"] - 1 
            if n_idx < self.max_nodes_pad:
                node_features[n_idx, 0] = node["P_mw"]
                node_features[n_idx, 1] = 1.0 if node.get("is_feeder_start", False) else 0.0
                
        # 展平特征，尺寸变为: 32 * 2 = 64
        flat_features = node_features.flatten()
        return flat_features, target_matrix, node_mask

# ==========================================
# 模块二：LLM 隐状态代理发生器 (Fake Qwen)
# ==========================================
class FakeQwenEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=4096):
        super().__init__()
        # 简单两层网络，把 64 维特征映射成 Qwen 的 4096 维隐藏状态
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

# ==========================================
# 模块三：连续化多任务预测头 (Topology Head)
# ==========================================
class TopologyHead(nn.Module):
    def __init__(self, hidden_dim=4096, max_nodes_pad=32):
        super().__init__()
        self.max_nodes_pad = max_nodes_pad
        self.matrix_size = max_nodes_pad * max_nodes_pad
        
        # 1. 降维全连接层
        self.projection = nn.Linear(hidden_dim, self.matrix_size)
        
        # 预注册对角线 Mask (不可训练的 buffer)
        mask = 1.0 - torch.eye(self.max_nodes_pad)
        self.register_buffer("diag_mask", mask)

    def forward(self, h):
        batch_size = h.size(0)
        
        # 第一步：降维与重塑 [batch, 4096] -> [batch, 32, 32]
        flat_matrix = self.projection(h)
        M = flat_matrix.view(batch_size, self.max_nodes_pad, self.max_nodes_pad)
        
        # 第二步：强制对称化 (无向图)
        M_sym = (M + M.transpose(-1, -2)) / 2.0
        
        # 第三步：概率压缩 (压在 0~1 之间)
        A_prob = torch.sigmoid(M_sym)
        
        # 第四步：对角线清零 (消除自环)
        A_final = A_prob * self.diag_mask
        
        return A_final

# ==========================================
# 模块四：数据流与前向传播测试验证
# ==========================================
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "Data/Dataset-2w-v1.jsonl" # 替换为你的真实路径
    MAX_NODES_PAD = 32
    HIDDEN_DIM = 4096
    BATCH_SIZE = 4
    
    print("🚀 开始初始化流水线...\n")
    
    # 1. 加载数据
    try:
        dataset = GridTopologyDataset(DATA_PATH, max_nodes_pad=MAX_NODES_PAD)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        flat_features, target_matrix, node_mask = next(iter(dataloader))
        print(f"✅ 数据加载成功! 获取 Batch Size: {BATCH_SIZE}")
        print(f"   输入特征形状: {flat_features.shape}")
        print(f"   目标矩阵形状: {target_matrix.shape}")
    except FileNotFoundError:
        print(f"❌ 找不到数据文件: {DATA_PATH}，请检查路径！")
        exit()

    # 2. 实例化模型
    fake_llm = FakeQwenEncoder(input_dim=MAX_NODES_PAD * 2, hidden_dim=HIDDEN_DIM)
    topo_head = TopologyHead(hidden_dim=HIDDEN_DIM, max_nodes_pad=MAX_NODES_PAD)
    
    # 3. 跑通前向传播 (Forward Pass)
    print("\n⚡ 正在执行前向传播...")
    h = fake_llm(flat_features)             # 大模型吐出隐藏状态
    pred_adj_matrix = topo_head(h)          # 拓扑头预测连通概率矩阵
    
    print(f"✅ 前向传播完成!")
    print(f"   大模型隐状态形状: {h.shape}")
    print(f"   预测连续矩阵形状: {pred_adj_matrix.shape}")
    print(f"   预测矩阵数值范围: [{pred_adj_matrix.min().item():.4f}, {pred_adj_matrix.max().item():.4f}] (应在 0~1 之间)")
    
    # 检查对角线是否真的被清零了
    diag_sum = torch.diagonal(pred_adj_matrix, dim1=-2, dim2=-1).sum().item()
    print(f"   对角线元素总和:   {diag_sum:.4f} (应为 0.0000)")


import torch.nn.functional as F
import torch.optim as optim

# ==========================================
# 模块五：代数图论物理裁决层 (Laplacian Loss)
# ==========================================
class LaplacianPhysicsLoss(nn.Module):
    def __init__(self, max_nodes_pad=32, physics_margin=0.1):
        super().__init__()
        self.max_nodes_pad = max_nodes_pad
        self.physics_margin = physics_margin # 物理安全裕度，要求 λ2 至少大于这个值
        
        # 预生成 Jitter 噪声矩阵 (极小的递增对角线)，用于防止梯度爆炸
        # 范围在 1e-5 到 1e-4 之间
        jitter_diag = torch.linspace(1e-5, 1e-4, max_nodes_pad)
        self.register_buffer("jitter_matrix", torch.diag(jitter_diag))

    def forward(self, pred_A, target_A, node_mask):
        """
        pred_A: 拓扑头预测的连续概率矩阵 [batch, 32, 32]
        target_A: 真实的 0/1 拓扑矩阵 [batch, 32, 32]
        node_mask: 节点掩码 [batch, 32]，1代表真实节点，0代表Padding的空节点
        """
        batch_size = pred_A.size(0)
        
        # -----------------------------------------
        # 1. 监督 Loss (BCE Loss)
        # -----------------------------------------
        # 生成 32x32 的二维掩码，屏蔽掉 Padding 节点之间的计算
        mask_2d = node_mask.unsqueeze(2) * node_mask.unsqueeze(1) 
        
        # 使用 BCE 计算预测矩阵和真实矩阵的差异
        bce_loss = F.binary_cross_entropy(pred_A, target_A, weight=mask_2d, reduction='sum') 
        bce_loss = bce_loss / mask_2d.sum().clamp(min=1.0) # 归一化
        
        # -----------------------------------------
        # 2. 物理 Loss (Laplacian Eigenvalue Loss)
        # -----------------------------------------
        physics_loss = 0.0
        
        # 由于 PyTorch 的 eigh 支持 batch 操作，我们可以批量算特征值！
        # a. 计算度矩阵 D (行求和)
        D_diag = torch.sum(pred_A, dim=-1)
        D = torch.diag_embed(D_diag)
        
        # b. 构造拉普拉斯矩阵 L = D - A
        L = D - pred_A
        
        # c. 注入保命 Jitter (极其重要，防 NaN 神器)
        L_safe = L + self.jitter_matrix.unsqueeze(0)
        
        # d. 求解特征值与特征向量 (eigh要求输入对称矩阵，返回升序排列的特征值)
        # eigenvalues shape: [batch, 32]
        eigenvalues, _ = torch.linalg.eigh(L_safe)
        
        # e. 提取 Fiedler Value (代数连通度 λ2，即第二小特征值)
        # eigenvalues[:, 0] 趋近于 0，eigenvalues[:, 1] 就是 λ2
        lambda_2 = eigenvalues[:, 1]
        
        # f. 计算物理惩罚: 如果 λ2 小于安全裕度，给予强力惩罚
        # ReLU(margin - lambda_2) 保证了只有当图濒临断裂时才产生 Loss
        physics_loss = F.relu(self.physics_margin - lambda_2).mean()

        return bce_loss, physics_loss, lambda_2.detach()


# ==========================================
# 终极测试：单批次过拟合炼丹炉 (Single Batch Overfit)
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "Data/Dataset-2w-v1.jsonl" 
    MAX_NODES_PAD = 32
    HIDDEN_DIM = 4096
    BATCH_SIZE = 4
    
    # 初始化数据和模型
    dataset = GridTopologyDataset(DATA_PATH, max_nodes_pad=MAX_NODES_PAD)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    flat_features, target_matrix, node_mask = next(iter(dataloader))
    
    fake_llm = FakeQwenEncoder(input_dim=MAX_NODES_PAD * 2, hidden_dim=HIDDEN_DIM)
    topo_head = TopologyHead(hidden_dim=HIDDEN_DIM, max_nodes_pad=MAX_NODES_PAD)
    loss_fn = LaplacianPhysicsLoss(max_nodes_pad=MAX_NODES_PAD, physics_margin=0.1)
    
    # 把 fake_llm 和 topo_head 的参数绑在一起给优化器
    optimizer = optim.Adam(list(fake_llm.parameters()) + list(topo_head.parameters()), lr=1e-3)
    
    print("🔥 开启物理法则注入炼丹炉 (单批次过拟合测试) 🔥\n")
    
    epochs = 100
    beta = 1.0 # 物理 Loss 的权重
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. 前向传播
        h = fake_llm(flat_features)
        pred_adj_matrix = topo_head(h)
        
        # 2. 物理裁决
        bce_loss, physics_loss, lambda_2 = loss_fn(pred_adj_matrix, target_matrix, node_mask)
        
        # 3. 联合优化
        total_loss = bce_loss + beta * physics_loss
        
        # 4. 反向传播
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch:3d}/{epochs}] | Total Loss: {total_loss.item():.4f} "
                  f"(BCE: {bce_loss.item():.4f}, Phys: {physics_loss.item():.4f}) | "
                  f"Avg λ2: {lambda_2.mean().item():.4f}")
            
    print("\n✅ 测试结束：如果 Loss 顺利降到接近 0，且没有报错 NaN，说明你的数学图谱已大功告成！")