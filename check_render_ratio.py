import re
import glob

def parse_file(filename):
    dynamic_values = []
    rigidity_ratios = []
    total_num = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # 匹配 Dynamic Cluster
            match_dynamic = re.search(r"^Dynamic Cluster:\s*(\d+)\s*/\s*(\d+)", line)
            if match_dynamic:
                x, y = map(int, match_dynamic.groups())
                dynamic_values.append(x)
                total_num = y

            # 匹配 Rigidity Dynamic Cluster
            match_rigidity = re.search(r"^Rigidity Dynamic Cluster:\s*(\d+)\s*/\s*(\d+)", line)
            if match_rigidity:
                a, b = map(int, match_rigidity.groups())
                if b != 0:
                    rigidity_ratios.append(a / b)

    avg_dynamic = sum(dynamic_values) / len(dynamic_values) if dynamic_values else 0
    avg_rigidity = sum(rigidity_ratios) / len(rigidity_ratios) if rigidity_ratios else 0

    return avg_dynamic, avg_rigidity, total_num


if __name__ == "__main__":
    # 假设要读取当前目录下的所有 .txt 文件
    files = glob.glob("/root/autodl-tmp/4DGaussiansOutput/*/*/run_render_ClusterSDMaskOnly.log")

    for file in files:
        avg_dynamic, avg_rigidity, total_num = parse_file(file)
        print(f"文件: {file}")
        print(f"  平均 Dynamic Cluster: {avg_dynamic:.2f}/ {total_num:.2f}")
        print(f"  平均 Rigidity Dynamic Cluster 比例: {avg_rigidity:.4f}")
        print()
