from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# 引入插值模式
from torchvision.transforms import InterpolationMode
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    scales = [1, 2, 4, 8] 

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                for scale in scales:
                    scale_name = f"x{scale}" if scale == 1 else f"x1_{scale}"
                    print(f"--- Evaluating at scale: {scale_name} ---")

                    full_dict[scene_dir][method][scale_name] = {}
                    per_view_dict[scene_dir][method][scale_name] = {}

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc=f"Metrics {scale_name}"):
                        B, C, H, W = renders[idx].shape
                        new_H, new_W = H // scale, W // scale
                        
                        if scale > 1:
                            # ================= 修改核心区域 =================
                            # Render: 使用 Nearest (最近邻) 模拟采样不足/锯齿，
                            # 这模拟了没有 mipmap 的 3DGS 在低分辨率下的表现。
                            cur_render = tf.resize(
                                renders[idx], 
                                [new_H, new_W], 
                                interpolation=InterpolationMode.NEAREST, 
                                antialias=False
                            )
                            
                            # GT: 使用 Bicubic + Antialias (抗锯齿) 模拟理想的低分辨率图像，
                            # 也就是人眼看到的平滑缩小效果。
                            cur_gt = tf.resize(
                                gts[idx], 
                                [new_H, new_W], 
                                interpolation=InterpolationMode.BICUBIC, 
                                antialias=True
                            )
                            # ===============================================
                        else:
                            cur_render = renders[idx]
                            cur_gt = gts[idx]

                        ssims.append(ssim(cur_render, cur_gt))
                        psnrs.append(psnr(cur_render, cur_gt))
                        lpipss.append(lpips_fn(cur_render, cur_gt).detach())

                    mean_ssim = torch.tensor(ssims).mean()
                    mean_psnr = torch.tensor(psnrs).mean()
                    mean_lpips = torch.tensor(lpipss).mean()

                    print(f"  [{scale_name}] SSIM : {mean_ssim:>12.7f}")
                    print(f"  [{scale_name}] PSNR : {mean_psnr:>12.7f}")
                    print(f"  [{scale_name}] LPIPS: {mean_lpips:>12.7f}")
                    print("")

                    full_dict[scene_dir][method][scale_name].update({
                        "SSIM": mean_ssim.item(),
                        "PSNR": mean_psnr.item(),
                        "LPIPS": mean_lpips.item()
                    })
                    per_view_dict[scene_dir][method][scale_name].update({
                        "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                    })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)