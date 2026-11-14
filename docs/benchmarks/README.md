# Benchmarks

## Template

Chạy các lệnh sau trên từng cấu hình hardware:

```bash
vipera profile > profile.json
vipera plan -m model.gguf --json > plan.json
vipera verify -m model.gguf --json > verify.json
```

Ghi rõ:
- GPU model, VRAM, driver version
- llama.cpp version
- OS + kernel version
- Model GGUF (arch, quant, size)

## Kết quả (điền sau khi chạy)

| GPU | Model | n_gpu_layers | Predicted VRAM | Actual VRAM | Predicted RAM | Actual RAM | Throughput | Error VRAM | Error RAM | Pass? |
|-----|-------|--------------|----------------|-------------|---------------|------------|------------|------------|-----------|-------|
| GTX 1650 4GB | Qwen2.5-1.5B-Q4_K | 14 | 3200 MB | ? | 2100 MB | ? | 6.5 t/s | ? | ? | ? |
