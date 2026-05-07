import random
import networkx as nx
import math
import pandapower as pp
import pandapower.networks as nw
from typing import Dict, Any, Optional, Tuple, List
import json
import argparse
import time
import swanlab  # 🟢 引入 SwanLab
from tqdm import tqdm

class TopologyGenerator:
    def __init__(self, min_nodes: int, max_nodes: int):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def generate(self, num_feeders: int) -> Tuple[Dict[str, Any], List[List[int]]]:
        while True:
            N = random.randint(self.min_nodes, self.max_nodes)
            G = self._build_graph(N, num_feeders)
            
            valid_switch_edges = [e for e in G.edges() 
                                  if G.nodes[e[0]]['type'] == 'Busbar' and G.nodes[e[1]]['type'] == 'Busbar']
            
            if len(valid_switch_edges) < num_feeders - 1:
                continue

            open_edges = None
            for _ in range(1000):
                candidate_edges = random.sample(valid_switch_edges, num_feeders - 1)
                if self._validate_topology(G, candidate_edges, num_feeders):
                    open_edges = candidate_edges
                    break
            
            if not open_edges:
                continue 
                
            topology_json, open_switches_json = self._format_output(G, open_edges)
            return topology_json, open_switches_json

    def _build_graph(self, N: int, k: int) -> nx.Graph:
        G = nx.Graph()
        for i in range(1, N + 1):
            G.add_node(i, type='Busbar')
            
        backbone_len = max(3, int(N * 0.6))
        backbone_nodes = list(range(1, backbone_len + 1))
        for i in range(backbone_len - 1):
            G.add_edge(backbone_nodes[i], backbone_nodes[i+1])
            
        leaves = [backbone_nodes[0], backbone_nodes[-1]]
        available_attach_points = backbone_nodes[1:-1]
        remaining_nodes = list(range(backbone_len + 1, N + 1))
        
        for _ in range(k - 2):
            if not remaining_nodes: break
            attach_point = random.choice(available_attach_points)
            branch_len = random.randint(1, max(1, len(remaining_nodes) - (k - 3 - _)))
            branch_nodes = remaining_nodes[:branch_len]
            remaining_nodes = remaining_nodes[branch_len:]
            
            G.add_edge(attach_point, branch_nodes[0])
            for i in range(branch_len - 1):
                G.add_edge(branch_nodes[i], branch_nodes[i+1])
                
            leaves.append(branch_nodes[-1])
            available_attach_points.extend(branch_nodes[:-1])
            
        for node in remaining_nodes:
            target_leaf = random.choice(leaves)
            G.add_edge(target_leaf, node)
            leaves.remove(target_leaf)
            leaves.append(node)
            available_attach_points.append(target_leaf)
            
        for i, leaf in enumerate(leaves[:k]):
            feeder_id = N + 1 + i
            G.add_node(feeder_id, type=f'Feeder_{i+1}')
            G.add_edge(leaf, feeder_id)
            
        return G

    def _validate_topology(self, G: nx.Graph, open_edges: List[Tuple[int, int]], num_feeders: int) -> bool:
        G_test = G.copy()
        G_test.remove_edges_from(open_edges)
        
        components = list(nx.connected_components(G_test))
        if len(components) != num_feeders:
            return False
            
        for comp in components:
            feeder_count = sum(1 for n in comp if str(G.nodes[n].get('type', '')).startswith('Feeder'))
            if feeder_count != 1:
                return False 
                
        return True

    def _format_output(self, G: nx.Graph, open_edges: List[Tuple[int, int]]) -> Tuple[Dict, List[List[int]]]:
        nodes_list = [{"node_id": n, "type": d["type"]} for n, d in G.nodes(data=True)]
        edges_list = [list(e) for e in G.edges()]
        
        Topology_Graph = {"nodes": nodes_list, "contact_lines": edges_list}
        open_switches = sorted([sorted(list(e)) for e in open_edges])
        return Topology_Graph, open_switches

class ParameterAllocator:
    def __init__(self, max_total_load: float, target_pf: float):
        self.max_total_load = max_total_load
        self.q_factor = math.tan(math.acos(target_pf))
        self.wire_types = {
            "LGJ-120": {"r0": 0.270, "x0": 0.335},
            "YJV22-300": {"r0": 0.078, "x0": 0.098}
        }

    def allocate(self, topology_graph: Dict, open_switches: List[List[int]]) -> Tuple[Dict, List[List[int]]]:
        graph_with_physicals = {"nodes": [], "edges": []}

        bus_nodes = [n for n in topology_graph["nodes"] if n["type"] == "Busbar"]
        raw_p = {n["node_id"]: random.uniform(0.1, 1.0) for n in bus_nodes}
        total_p = sum(raw_p.values())
        
        scale_factor = 1.0
        if total_p > self.max_total_load:
            scale_factor = (self.max_total_load - 1.0) / total_p

        for node in topology_graph["nodes"]:
            new_node = node.copy()
            if node["type"] == "Busbar":
                p_val = raw_p[node["node_id"]] * scale_factor
                new_node["P_load"] = round(p_val, 4)
                new_node["Q_load"] = round(p_val * self.q_factor, 4)
            else:
                new_node["P_load"], new_node["Q_load"] = 0.0, 0.0
                new_node["bus_type"] = "Slack"
            graph_with_physicals["nodes"].append(new_node)

        for edge in topology_graph["contact_lines"]:
            wire_name = random.choice(list(self.wire_types.keys()))
            length_km = random.uniform(0.5, 3.0)
            r_val = self.wire_types[wire_name]["r0"] * length_km
            x_val = self.wire_types[wire_name]["x0"] * length_km
            
            graph_with_physicals["edges"].append({
                "from": edge[0], "to": edge[1],
                "length_km": round(length_km, 3),
                "R": round(r_val, 4), "X": round(x_val, 4)
            })

        return graph_with_physicals, open_switches

class PowerFlowEngine:
    def __init__(self, base_kv: float, vm_pu: float):
        self.base_kv = base_kv
        self.vm_pu = vm_pu

    def run_simulation(self, graph_with_physicals: Dict, open_switches: List[List[int]], num_feeders: int) -> Optional[Dict]:
        net = pp.create_empty_network()
        bus_mapping = {}

        for node in graph_with_physicals["nodes"]:
            bus_idx = pp.create_bus(net, vn_kv=self.base_kv, name=f"{node['type']}_{node['node_id']}")
            bus_mapping[node["node_id"]] = bus_idx

            if node["type"] == "Busbar":
                pp.create_load(net, bus=bus_idx, p_mw=node["P_load"], q_mvar=node["Q_load"])
            elif "Feeder" in node["type"]:
                pp.create_ext_grid(net, bus=bus_idx, vm_pu=self.vm_pu, name=node["type"])

        open_switch_sets = [set(edge) for edge in open_switches]

        for edge in graph_with_physicals["edges"]:
            current_pair = {edge["from"], edge["to"]}
            is_in_service = current_pair not in open_switch_sets

            r_ohm_per_km = edge["R"] / edge["length_km"]
            x_ohm_per_km = edge["X"] / edge["length_km"]

            pp.create_line_from_parameters(
                net, from_bus=bus_mapping[edge["from"]], to_bus=bus_mapping[edge["to"]], 
                length_km=edge["length_km"], r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km, 
                c_nf_per_km=0.0, max_i_ka=10.0, in_service=is_in_service
            )

        try:
            pp.runpp(net, algorithm='nr', enforce_q_lims=False)
        except Exception:
            return None

        power_flow_result = {f"Feeder_{i+1}_P_Inject_MW": 0.0 for i in range(num_feeders)}
        for idx in net.ext_grid.index:
            bus_idx = net.ext_grid.at[idx, "bus"]
            bus_name = net.bus.at[bus_idx, "name"]
            p_mw = net.res_ext_grid.at[idx, "p_mw"]
            
            for i in range(num_feeders):
                if f"Feeder_{i+1}" in bus_name:
                    power_flow_result[f"Feeder_{i+1}_P_Inject_MW"] = round(p_mw, 4)

        return power_flow_result

class DatasetFormatter:
    def __init__(self, max_nodes_pad: int, round_decimals: int = 2):
        self.round_decimals = round_decimals
        self.max_nodes_pad = max_nodes_pad 
        self.system_prompt = (
            "【角色设定】\n"
            "你是一位顶级的电力系统调度专家，擅长通过严密的图论与潮流分析，进行配电网最优拓扑重构。\n\n"
            "【物理定律与安全硬约束（绝对不可违反）】\n"
            "1. 辐射状拓扑原则：配电网必须保持树状（辐射状）结构运行。任意一个负荷节点，必须有且仅有一条路径连通至唯一的电源点。\n"
            "2. 严禁短路/电磁环网（致命错误）：如果网络中存在一条通路，将两个不同的电源点直接连接，或者形成闭合的环路，将导致极大的短路电流或环流，烧毁设备。必须通过“断开联络线（开环）”来消除所有环路。\n"
            "3. 严禁孤岛停电（致命错误）：如果断开了错误的联络线，导致某些负荷节点与所有电源点彻底断开连接，将形成“孤岛”，造成大面积停电事故。\n\n"
            "【任务要求】\n"
            "- 分析目标：给定多电源配电网的拓扑与量测数据，请通过数值计算与逻辑推理，准确判断开环点位于哪几条联络线。\n"
            "- 优化目标：在绝对满足上述“无环流、无孤岛”硬约束的前提下，尽可能使各侧电源承担的有功功率 (P_mw) 均衡，降低整体网损。\n"
            "- 输出格式：必须先在 <think> ... </think> 标签内进行严密的拓扑与功率推理，最后输出所有开环点所在线路（两个母线的id），格式必须为包含多个数组的二维数组，如 [[1, 2], [3, 4]]。\n\n"
            "【配电网的拓扑数据，以json格式描述】"
        )

    def check_fatal_errors(self, all_edges, false_switches, feeders):
        G_test = nx.Graph()
        G_test.add_edges_from(all_edges)
        for u, v in false_switches:
            if G_test.has_edge(u, v): G_test.remove_edge(u, v)
            elif G_test.has_edge(v, u): G_test.remove_edge(v, u)

        components = list(nx.connected_components(G_test))
        island_nodes = []
        short_feeders = []

        for comp in components:
            comp_feeders = [n for n in comp if n in feeders]
            if len(comp_feeders) == 0:
                island_nodes.extend([n for n in comp]) 
            elif len(comp_feeders) > 1:
                short_feeders.append(comp_feeders) 

        return island_nodes, short_feeders

    def format_sample(self, graph_with_physicals, power_flow_result, open_switches, scenario, num_feeders):
        formatted_nodes = []
        feeders = [] 
        for node in graph_with_physicals["nodes"]:
            node_type = node["type"]
            if "Feeder" in node_type:
                p_val = power_flow_result[f"{node_type}_P_Inject_MW"]
                is_feeder = True
                feeders.append(node["node_id"])
            else:
                p_val = node.get("P_load", 0.0)
                is_feeder = False
                
            formatted_node = {
                "node_id": node["node_id"],
                "P_mw": round(p_val, self.round_decimals)
            }
            if is_feeder:
                formatted_node["is_feeder_start"] = True
            formatted_nodes.append(formatted_node)
            
        formatted_contact_lines = [list(e) for e in [tuple(sorted([edge["from"], edge["to"]])) for edge in graph_with_physicals["edges"]]]
        true_switches = sorted(open_switches)

        target_adj_matrix = [[0] * self.max_nodes_pad for _ in range(self.max_nodes_pad)]
        node_mask = [0] * self.max_nodes_pad
        
        for node in graph_with_physicals["nodes"]:
            idx = node["node_id"] - 1
            if idx < self.max_nodes_pad:
                node_mask[idx] = 1
                
        open_switches_set = set([tuple(sorted(e)) for e in open_switches])
        for edge in formatted_contact_lines:
            u, v = edge[0] - 1, edge[1] - 1
            if tuple(sorted(edge)) not in open_switches_set:
                if u < self.max_nodes_pad and v < self.max_nodes_pad:
                    target_adj_matrix[u][v] = 1
                    target_adj_matrix[v][u] = 1 

        user_input_dict = {
            "nodes": formatted_nodes,
            "contact_lines": formatted_contact_lines,
            "target_adj_matrix": target_adj_matrix, 
            "node_mask": node_mask                  
        }

        if scenario == 1:
            assistant_response = f"""<think>
【步骤1】拓扑解析与安全性约束：
- 目标：在网络中寻找合法的联络线进行开环，消除所有环流。
- 尝试方案：切断线路 {true_switches}。经图论连通性分析，切断该组线路后，全网无闭合环路（防短路），且所有负荷节点均连通至唯一电源（防孤岛）。满足安全底线。
【步骤2】潮流计算与全局寻优：
- 结合各节点的注入有功功率 (P_mw) 数据，该断开方案能够使各电源侧承担的负荷最为均衡，避免单侧重载，网损极小。
- 结论：此方案为满足安全约束下的全局最优解。
</think>

{true_switches}"""
            
        elif scenario == 2:
            user_input_dict["reported_open_switches"] = true_switches
            assistant_response = f"""<think>
【步骤1】参考方案核查：
- 接收到上报的开环方案：{true_switches}。
- 拓扑校验：经校验，切断上述线路后，电网呈标准辐射状，未产生断电孤岛，未发生多电源短路，拓扑结构绝对合法。
【步骤2】功率优度评估：
- 评估该方案下的 P_mw 分布，各电源出力相对均衡，无严重偏载与网损异常。
- 结论：上报方案既安全且最优，予以直接采纳。
</think>

{true_switches}"""
            
        elif scenario == 3:
            all_edges = sorted(formatted_contact_lines)
            max_attempts = 50
            fatal_found = False
            
            for _ in range(max_attempts):
                keep_count = random.randint(0, max(0, num_feeders - 2))
                kept_switches = random.sample(true_switches, keep_count) if keep_count > 0 else []
                available_false = [e for e in all_edges if e not in true_switches]
                needed_false_count = (num_feeders - 1) - keep_count
                
                if len(available_false) >= needed_false_count:
                    added_false = random.sample(available_false, needed_false_count)
                    false_switches = sorted(kept_switches + added_false)
                else:
                    false_switches = true_switches
                    break
                
                island_nodes, short_feeders = self.check_fatal_errors(all_edges, false_switches, feeders)
                
                if len(island_nodes) > 0 or len(short_feeders) > 0:
                    fatal_found = True
                    break
                
                if num_feeders == 2:
                    break

            user_input_dict["reported_open_switches"] = false_switches
            
            if fatal_found:
                error_diagnosis = ""
                if island_nodes:
                    display_islands = island_nodes[:3]
                    etc = " 等" if len(island_nodes) > 3 else ""
                    error_diagnosis += f"导致节点 {display_islands}{etc} 失去全部电源供应形成失电孤岛；"
                if short_feeders:
                    error_diagnosis += f"未能彻底解环，导致电源之间形成致命的短路环流；"

                assistant_response = f"""<think>
【步骤1】上报方案风险驳回：
- 接收到待审核开环方案：{false_switches}。
- 拓扑安全分析：经连通性计算，若强行切断该组线路，将{error_diagnosis}
- 结论：该方案严重违反《电网安全法典》的安全底线，属于灾难性物理违规，必须予以坚决驳回！
【步骤2】重新计算全局最优解：
- 重新对网络结构与节点 P_mw 进行测算，寻找唯一合法的最佳切断点。
- 正确方案应为切断 {true_switches}。
</think>

{true_switches}"""
            else:
                assistant_response = f"""<think>
【步骤1】上报方案风险驳回：
- 接收到待审核开环方案：{false_switches}。
- 拓扑安全分析：经连通性计算，该方案虽未造成孤岛或短路，满足基本的拓扑连通底线。
- 潮流优度分析：然而，该方案完全无视了 P_mw 数据的全局分布，切断该位置会导致两侧电源的功率分配严重失衡，引发局部线路重载与网损剧增！
- 结论：该方案为典型的功率劣质解，予以驳回！
【步骤2】重新计算全局最优解：
- 重新对网络结构与 P_mw 进行测算，必须寻找使负载最均衡的最佳切断点。
- 正确且最优的开环方案应为切断 {true_switches}。
</think>

{true_switches}"""

        topology_json_str = json.dumps(user_input_dict, ensure_ascii=False)
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": topology_json_str},
                {"role": "assistant", "content": assistant_response}
            ]
        }

class PowerGridDataPipeline:
    def __init__(self, config: argparse.Namespace):
        self.generator = TopologyGenerator(min_nodes=config.min_nodes, max_nodes=config.max_nodes)
        self.allocator = ParameterAllocator(max_total_load=config.max_total_load, target_pf=config.target_pf)
        self.engine = PowerFlowEngine(base_kv=config.base_kv, vm_pu=config.vm_pu)
        self.formatter = DatasetFormatter(max_nodes_pad=config.max_nodes_pad)
        self.config = config

    def generate_batch(self):
        target_sample_count = self.config.samples
        output_filename = self.config.output if self.config.output else f"grid_data_{target_sample_count}.jsonl"
        
        feeder_choices = [2, 3, 4]
        scenario_choices = [1, 2, 3]
        
        valid_samples = 0
        feeder_counts = {2: 0, 3: 0, 4: 0}
        scenario_counts = {1: 0, 2: 0, 3: 0}
        
        start_time = time.time()
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            with tqdm(total=target_sample_count, desc="生成进度") as pbar:
                while valid_samples < target_sample_count:
                    num_feeders = random.choices(feeder_choices, weights=self.config.feeders_weight, k=1)[0]
                    
                    topo, switches = self.generator.generate(num_feeders)
                    physicals, switches = self.allocator.allocate(topo, switches)
                    pf_results = self.engine.run_simulation(physicals, switches, num_feeders)
                    
                    if pf_results is None:
                        continue
                        
                    scenario = random.choices(scenario_choices, weights=self.config.scenarios_weight, k=1)[0]
                    
                    feeder_counts[num_feeders] += 1
                    scenario_counts[scenario] += 1
                    
                    final_sample = self.formatter.format_sample(physicals, pf_results, switches, scenario, num_feeders)
                    f.write(json.dumps(final_sample, ensure_ascii=False) + '\n')
                    
                    valid_samples += 1
                    pbar.update(1)
                    
                    # 🟢 阶段性向 SwanLab 推送数据 (每50条推一次，避免 IO 阻塞)
                    if valid_samples % 50 == 0:
                        swanlab.log({"progress/valid_samples": valid_samples}, step=valid_samples)
                        
        # 🟢 任务结束，汇总最终分布数据推送到 SwanLab
        end_time = time.time()
        swanlab.log({
            "distribution/feeder_2": feeder_counts[2],
            "distribution/feeder_3": feeder_counts[3],
            "distribution/feeder_4": feeder_counts[4],
            "distribution/scenario_blind": scenario_counts[1],
            "distribution/scenario_compliant": scenario_counts[2],
            "distribution/scenario_error": scenario_counts[3],
            "metrics/total_time_seconds": round(end_time - start_time, 2)
        })
                    
        print(f"\n✅ 数据集生成完毕！已保存至 {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多联络配电网数据集生成器")
    
    # ---------------- 🟢 SwanLab 与环境配置 ----------------
    parser.add_argument("--project", type=str, default="PowerGrid-Topology", help="SwanLab 项目名称")
    parser.add_argument("--exp_name", type=str, default=None, help="SwanLab 实验名称 (默认根据参数自动生成)")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子 (保证数据绝对可复现)")
    
    # ---------------- 核心控制参数 ----------------
    parser.add_argument("--samples", type=int, default=100, help="目标生成样本数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件名")
    
    # ---------------- 拓扑与填充超参数 ----------------
    parser.add_argument("--min_nodes", type=int, default=15, help="系统母线的最小节点数")
    parser.add_argument("--max_nodes", type=int, default=26, help="系统母线的最大节点数")
    parser.add_argument("--max_nodes_pad", type=int, default=32, help="神经网络右脑预测的固定矩阵维度大小")
    
    # ---------------- 电气模拟超参数 ----------------
    parser.add_argument("--max_total_load", type=float, default=12.0, help="全局总负荷 MW 限制")
    parser.add_argument("--target_pf", type=float, default=0.90, help="负荷功率因数")
    parser.add_argument("--base_kv", type=float, default=10.0, help="基准电压 kV")
    parser.add_argument("--vm_pu", type=float, default=1.05, help="电源节点电压标幺值")
    
    # ---------------- 分布权重配置 ----------------
    parser.add_argument("--feeders_weight", type=float, nargs=3, default=[65.0, 20.0, 15.0], help="2,3,4电源权重")
    parser.add_argument("--scenarios_weight", type=float, nargs=3, default=[0.25, 0.60, 0.15], help="场景1,2,3权重")
    
    args = parser.parse_args()

    # 🟢 1. 设定全局随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🌱 已锁定全局随机种子: {args.seed}")
    else:
        print("⚠️ 未设定随机种子，将生成完全随机的数据集")

    # 简单安全校验
    assert args.max_nodes_pad >= args.max_nodes + 4, "❌ max_nodes_pad 必须大于 (max_nodes + 4个电源)"

    # 🟢 2. 初始化 SwanLab 实验
    exp_name = args.exp_name if args.exp_name else f"Gen-{args.samples}-Seed-{args.seed}"
    swanlab.init(
        project=args.project,
        name=exp_name,
        workspace="alex--",
        config=vars(args),  # 将所有 argparse 的参数自动打包保存为超参数看板
        description="配电网多联络数据集生成与拓扑矩阵提取"
    )

    # 3. 启动流水线
    pipeline = PowerGridDataPipeline(args)
    pipeline.generate_batch()
    
    # 🟢 4. 结束 SwanLab 实验
    swanlab.finish()