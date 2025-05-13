from pathlib import Path

import torch
from run_demo import PairformerTorch, PairformerTRT, create_original_model


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    original_model, _ = create_original_model(device)
    torch_module = PairformerTorch(original_model.pairformer_module)

    trt_module = PairformerTRT(
        Path("structure_pairformer_1_1_64_768_engines_layer_2"), 1, 0)

    s = torch.load("s_2.pt").to(device).contiguous()
    z = torch.load("z_2.pt").to(device).contiguous()
    mask = torch.load("mask_2.pt").to(device).contiguous()
    pair_mask = torch.load("pair_mask_2.pt").to(device).contiguous()

    with torch.no_grad():
        torch_output_s, torch_output_z = torch_module(s, z, mask, pair_mask)
        torch.cuda.synchronize()
        trt_output_s, trt_output_z = trt_module(s, z, mask, pair_mask)
        torch.cuda.synchronize()

    print(torch_output_s[0, 0, :100])
    print(trt_output_s[0, 0, :100])

    print(torch_output_z[0, 0, :100])
    print(trt_output_z[0, 0, :100])


if __name__ == "__main__":
    main()
