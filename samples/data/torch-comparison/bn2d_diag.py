import numpy as np, torch, torch.nn as nn

test_dir = r'E:\khurram-uworx\Nivara\samples\NivaraInference\Python\test'
inp = np.fromfile(f'{test_dir}/bn2d_16ch_input.bin', dtype=np.float32).reshape(1, 16, 5, 5)
gamma = np.fromfile(f'{test_dir}/bn2d_16ch_gamma.bin', dtype=np.float32)
beta = np.fromfile(f'{test_dir}/bn2d_16ch_beta.bin', dtype=np.float32)
rm = np.fromfile(f'{test_dir}/bn2d_16ch_running_mean.bin', dtype=np.float32)
rv = np.fromfile(f'{test_dir}/bn2d_16ch_running_var.bin', dtype=np.float32)

# Manual computation matching C# ForwardEval logic
eps = 1e-5
n, c, h, w = inp.shape
hw = h * w
inp_flat = inp.flatten()

output = np.zeros_like(inp_flat)
xhat = np.zeros_like(inp_flat)

for ch in range(c):
    m = rm[ch]
    inv = 1.0 / np.sqrt(rv[ch] + eps)
    g = gamma[ch]
    b = beta[ch]
    for i in range(n):
        offset = i * c * hw + ch * hw
        for j in range(hw):
            normalized = (inp_flat[offset + j] - m) * inv
            xhat[offset + j] = normalized
            output[offset + j] = normalized * g + b

expected = np.fromfile(f'{test_dir}/bn2d_16ch_output.bin', dtype=np.float32)
diff = np.abs(output - expected)
print(f'Manual C# logic vs expected: max diff = {diff.max()}, first 5 output: {output[:5]}, expected: {expected[:5]}')

# Also check: what does PyTorch actually compute?
bn = nn.BatchNorm2d(16)
bn.weight.data.copy_(torch.from_numpy(gamma))
bn.bias.data.copy_(torch.from_numpy(beta))
bn.running_mean.copy_(torch.from_numpy(rm))
bn.running_var.copy_(torch.from_numpy(rv))
bn.eval()
with torch.no_grad():
    out = bn(torch.from_numpy(inp)).numpy().flatten()
print(f'PyTorch eval output first 5: {out[:5]}')
print(f'PyTorch vs expected: max diff = {np.abs(out - expected).max()}')

# Check if PyTorch uses float32 throughout
print(f'PyTorch output dtype: {out.dtype}')
print(f'expected dtype: {expected.dtype}')

# Check for channel-first layout issue
# PyTorch output is [N,C,H,W], our manual is [N*C*H*W] flat
out_chw = out.flatten()
print(f'PyTorch flat first 5: {out_chw[:5]}')
print(f'Match: {np.allclose(out_chw, expected, atol=1e-5)}')
