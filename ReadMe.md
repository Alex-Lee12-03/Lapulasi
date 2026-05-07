### **生成数据流**

```bash
python run/data_generation.py \
    --samples 20000 \
    --project "LaPuLaSi-Data" \
    --exp_name "Dataset-2w-v1" \
    --seed 16 \
    --min_nodes 8 \
    --max_nodes 28 \
    --max_nodes_pad 32 \
    --max_total_load 15.0 \
    --output Data/Dataset-2w-v1.jsonl
	

```

