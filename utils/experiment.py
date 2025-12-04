

def build_experiment_name(
    model_name: str,
    head_type: str,
    side: str,
    mirror: bool,
    input_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    split_ratio: float | None = None,
    sigma: float | None = None,
) -> str:
    """
    統一產生實驗 / 檔名用的 base name，不含副檔名。
    規則：
      simcc: model_simcc_sr2.0_sigma6.0_cropleft[_mirror]_448_300_0.0001_32
      direct_regression: model_direct_regression_cropleft[_mirror]_448_300_0.0001_32
    """
    base = f"{model_name}_{head_type}"

    # 只有 simcc 系列 head 才會加 sr 跟 sigma
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"] and split_ratio is not None:
        base += f"_sr{split_ratio}"
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"] and sigma is not None:
        base += f"_sigma{sigma}"

    base += f"_crop{side}"
    if mirror:
        base += "_mirror"

    # 統一後面這四個
    base += f"_{input_size}_{epochs}_{learning_rate}_{batch_size}"
    return base