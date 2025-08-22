import re
import csv

def extract_metrics_from_file(input_file, output_file):
    # 用于匹配场景名（如flame_steak_Vanilla）和对应的SSIM、PSNR、LPIPS-vgg值的正则表达式
    scene_pattern = r"Scene:\s*(.*?)(?=\sSSIM)"
    ssim_pattern = r"SSIM\s*:\s*(\d+\.\d+)"
    psnr_pattern = r"PSNR\s*:\s*(\d+\.\d+)"
    lpips_pattern = r"LPIPS-vgg:\s*(\d+\.\d+)"

    # 存储结果
    data = {}

    with open(input_file, 'r') as file:
        content = file.read()

        # 提取所有场景名
        scenes = re.findall(scene_pattern, content)
        for scene in scenes:
            # 提取当前场景的SSIM、PSNR和LPIPS-vgg值
            ssim = re.search(ssim_pattern, content.split(scene)[-1])
            psnr = re.search(psnr_pattern, content.split(scene)[-1])
            lpips = re.search(lpips_pattern, content.split(scene)[-1])

            if ssim and psnr and lpips:
                # 以场景名为键，将SSIM、PSNR、LPIPS-vgg值存储为列表
                data[scene.strip()] = [ssim.group(1), psnr.group(1), lpips.group(1)]

    # 获取所有场景名作为横栏
    scenes_list = list(data.keys())

    # 输出到CSV文件，横栏是场景，纵栏是SSIM、PSNR、LPIPS-vgg
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入纵栏标题（SSIM, PSNR, LPIPS-vgg）
        writer.writerow(["Metric"] + scenes_list)

        # 写入每个指标（SSIM, PSNR, LPIPS-vgg）对应的值
        writer.writerow(["SSIM"] + [data[scene][0] for scene in scenes_list])
        writer.writerow(["PSNR"] + [data[scene][1] for scene in scenes_list])
        writer.writerow(["LPIPS-vgg"] + [data[scene][2] for scene in scenes_list])

    print(f"数据已成功写入到 {output_file}")

# 设置输入文件和输出文件的路径
input_file = 'tmp.txt'  # 需要处理的文本文件路径
output_file = 'output_metrics.csv'  # 输出的CSV文件路径

# 调用函数处理文件
extract_metrics_from_file(input_file, output_file)
