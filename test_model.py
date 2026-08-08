import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from torchinfo import summary
from models.model import initialize_model
from ultralytics import YOLO
from fvcore.nn import FlopCountAnalysis, parameter_count

if __name__ == "__main__":
    model = initialize_model("convnext_tiny_fpn1234concat", num_points=4, head_type="simcc_2d", input_size=(224, 224), Nx=672, Ny=672)
    # summary(model, input_size=(1, 3, 224, 224))
    
    # 2. 準備假輸入 (Dummy Input)
    # 請務必填入「實際推論時」的長寬，因為運算量(FLOPs)會隨長寬平方倍增
    dummy_input = torch.randn(1, 3, 224, 224)

    # ==========================================
    # 3. 計算精確的運算量 (MACs)
    # ==========================================
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False) # 關閉警告讓終端機乾淨一點
    total_macs = flops.total()

    # ==========================================
    # 4. 計算精確的參數量 (Parameters)
    # ==========================================
    # fvcore 會回傳一個字典，包含各層的參數量，其中 key 為空字串 "" 的就是總參數量
    params_dict = parameter_count(model)
    total_params = params_dict[""]

    # ==========================================
    # 5. 輸出精確結果 (包含千分位整數與易讀單位)
    # ==========================================
    print("="*60)
    print(f"精確參數量 (Parameters) : {total_params:,} (約 {total_params / 1e6:.2f} M)")
    print(f"精確運算量 (MACs)       : {total_macs:,} (約 {total_macs / 1e9:.3f} GMACs)")
    print(f"精確浮點運算 (FLOPs)    : {total_macs * 2:,} (約 {(total_macs * 2) / 1e9:.3f} GFLOPs)")
    print("="*60)
    
    # model = initialize_model("hrnet_w32", num_points=8, head_type="heatmap", input_size=(224, 224), Nx=672, Ny=672)
    # summary(model, input_size=(1, 3, 224, 224))
    
    # model = YOLO('weights/yolo26s_mtddh_set.pt')
    # model.info(detailed=True)
    
    # model = YOLO('hip-yolo-xray-seg.pt')
    # model.info(imgsz=800)