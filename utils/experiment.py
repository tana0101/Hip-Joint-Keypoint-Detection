def build_experiment_name(
    model_name: str,
    head_type: str,
    input_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    side: str | None = None,  # 修改：設為可選參數，單階段時不傳入或傳 None
    mirror: bool = False,     # 修改：設為預設 False
    split_ratio: float | None = None,
    sigma: float | None = None,
) -> str:
    """
    統一產生實驗 / 檔名用的 base name，不含副檔名。
    規則：
      【雙階段】: model_simcc_sr2.0_sigma6.0_cropleft_mirror_448_300_0.0001_32
      【單階段】: model_simcc_sr2.0_sigma6.0_onestage_448_300_0.0001_32
    """
    base = f"{model_name}_{head_type}"

    # 只有 simcc 系列 head 才會加 sr 跟 sigma
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"] and split_ratio is not None:
        base += f"_sr{split_ratio}"
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"] and sigma is not None:
        base += f"_sigma{sigma}"

    # === 區分單階段與雙階段的檔名 ===
    if side is not None:
        # 雙階段 (有指定 side)
        base += f"_crop{side}"
        if mirror:
            base += "_mirror"
    else:
        # 單階段 (沒有指定 side)
        base += "_onestage"

    # 統一後面這四個超參數
    base += f"_{input_size}_{epochs}_{learning_rate}_{batch_size}"
    return base