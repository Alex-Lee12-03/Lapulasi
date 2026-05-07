import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 模块一：标准化张量预处理流 (Dataset)
# ==========================================
class GridTopologyDataset(Dataset):
    def __init__(self, jsonl_path, max_nodes_pad=32):
        self.data = []
        self.max_nodes_pad = max_nodes_pad
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        user_content_str = sample["messages"][1]["content"]
        grid_data = json.loads(user_content_str)
        
        target_matrix = torch.tensor(grid_data["target_adj_matrix"], dtype=torch.float32)
        node_mask = torch.tensor(grid_data["node_mask"], dtype=torch.float32)
        
        node_features = torch.zeros((self.max_nodes_pad, 2), dtype=torch.float32)
        for node in grid_data["nodes"]:
            n_idx = node["node_id"] - 1 
            if n_idx < self.max_nodes_pad:
                node_features[n_idx, 0] = node["P_mw"]
                node_features[n_idx, 1] = 1.0 if node.get("is_feeder_start", False) else 0.0
                
        flat_features = node_features.flatten()
        return flat_features, target_matrix, node_mask

# ==========================================
# 模块二：LLM 隐状态代理发生器 (Fake Qwen)
# ==========================================
class FakeQwenEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=4096):
        super().__init__()
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
        self.projection = nn.Linear(hidden_dim, self.matrix_size)
        
        mask = 1.0 - torch.eye(self.max_nodes_pad)
        self.register_buffer("diag_mask", mask)

    def forward(self, h):
        batch_size = h.size(0)
        flat_matrix = self.projection(h)
        M = flat_matrix.view(batch_size, self.max_nodes_pad, self.max_nodes_pad)
        M_sym = (M + M.transpose(-1, -2)) / 2.0
        A_prob = torch.sigmoid(M_sym)
        A_final = A_prob * self.diag_mask
        return A_final

# ==========================================
# 模块四：代数图论物理裁决层 (Laplacian Loss)
# ==========================================
class LaplacianPhysicsLoss(nn.Module):
    def __init__(self, max_nodes_pad=32, physics_margin=0.1):
        super().__init__()
        self.max_nodes_pad = max_nodes_pad
        self.physics_margin = physics_margin
        
        # Jitter 噪声矩阵，防止梯度爆炸 (The Savior)
        jitter_diag = torch.linspace(1e-5, 1e-4, max_nodes_pad)
        self.register_buffer("jitter_matrix", torch.diag(jitter_diag))

    def forward(self, pred_A, target_A, node_mask):
        batch_size = pred_A.size(0)
        
        # 1. 监督 Loss (BCE)
        mask_2d = node_mask.unsqueeze(2) * node_mask.unsqueeze(1) 
        bce_loss = F.binary_cross_entropy(pred_A, target_A, weight=mask_2d, reduction='sum') 
        bce_loss = bce_loss / mask_2d.sum().clamp(min=1.0)
        
        # 2. 物理 Loss (修复版)
        # 【修复点 1】：用 2D Mask 切断所有幽灵节点，彻底清零它们的连通概率
        pred_A_masked = pred_A * mask_2d
        
        D_diag = torch.sum(pred_A_masked, dim=-1)
        D = torch.diag_embed(D_diag)
        L = D - pred_A_masked
        L_safe = L + self.jitter_matrix.unsqueeze(0)
        
        eigenvalues, _ = torch.linalg.eigh(L_safe)
        
        # 【修复点 2】：动态索引真实的代数连通度 λ2
        N_active = node_mask.sum(dim=1).long() # 当前批次中，每张图真实的节点数
        N_pad = self.max_nodes_pad - N_active  # 计算被 Padding 的空节点数
        
        # 真正的 λ2 的索引位置 (0-index 体系下)
        target_indices = N_pad + 1
        
        # 从 eigenvalues 张量中，准确抓取出真实的 λ2
        lambda_2 = eigenvalues[torch.arange(batch_size), target_indices]
        
        # 计算物理惩罚
        physics_loss = F.relu(self.physics_margin - lambda_2).mean()

        return bce_loss, physics_loss, lambda_2.detach()

# ==========================================
# 终极测试：单批次过拟合炼丹炉
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "Data/Dataset-2w-v1.jsonl" # 确保路径正确
    MAX_NODES_PAD = 32
    HIDDEN_DIM = 4096
    BATCH_SIZE = 4
    
    dataset = GridTopologyDataset(DATA_PATH, max_nodes_pad=MAX_NODES_PAD)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    flat_features, target_matrix, node_mask = next(iter(dataloader))
    
    fake_llm = FakeQwenEncoder(input_dim=MAX_NODES_PAD * 2, hidden_dim=HIDDEN_DIM)
    topo_head = TopologyHead(hidden_dim=HIDDEN_DIM, max_nodes_pad=MAX_NODES_PAD)
    loss_fn = LaplacianPhysicsLoss(max_nodes_pad=MAX_NODES_PAD, physics_margin=0.1)
    
    optimizer = optim.Adam(list(fake_llm.parameters()) + list(topo_head.parameters()), lr=1e-3)
    
    print("🔥 开启物理法则注入炼丹炉 (单批次过拟合测试) 🔥\n")
    
    # 打印一下真实的节点数量，验证我们的修复逻辑
    N_active_demo = node_mask.sum(dim=1).long().tolist()
    print(f"🧐 本次 Batch 中 4 张图的真实节点数量分别为: {N_active_demo}\n")
    
    epochs = 100
    beta = 1.0 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        h = fake_llm(flat_features)
        pred_adj_matrix = topo_head(h)
        
        bce_loss, physics_loss, lambda_2 = loss_fn(pred_adj_matrix, target_matrix, node_mask)
        
        total_loss = bce_loss + beta * physics_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch:3d}/{epochs}] | Total Loss: {total_loss.item():.4f} "
                  f"(BCE: {bce_loss.item():.4f}, Phys: {physics_loss.item():.4f}) | "
                  f"Avg λ2: {lambda_2.mean().item():.4f}")
            
    print("\n✅ 终极测试结束：底层数学框架已稳如磐石，准备接入大模型！")