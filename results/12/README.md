測試 300 epochs 在 fc 層上的微調狀況以及不同 batch size 上的差異
1. 2048個神經元
```
model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, POINTS_COUNT * 2)
        )
```
2. 直接將輸出層改為 POINTS_COUNT * 2
```
model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, POINTS_COUNT * 2)
        )
```