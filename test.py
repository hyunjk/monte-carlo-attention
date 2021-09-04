import torch

window_size = 5
num_head = 2
seq_length = 15
global_size = 2

total_size = window_size * 2 + 1 + global_size

v = torch.zeros((num_head, seq_length, total_size))

for h in range(num_head):
    for i in range(seq_length):
        y_start = i - window_size
        count = 0
        for j in range(window_size * 2 + 1):
            y = y_start + j
            if y < 0:
                continue
            if y >= seq_length:
                break

            x = window_size * 2 + global_size - j
            v[h][y][x] = (i + 1) + h * 100
            count += 1

vv = v.narrow(2, global_size, total_size - global_size)

print(v)
print(vv)

vv = vv.contiguous()
zz = torch.zeros((num_head, window_size, window_size * 2 + 1))
vv = torch.cat((zz, vv, zz), dim=1)
print(vv)

vvv = vv.as_strided((num_head, seq_length, window_size * 2 + 1),
                    ((seq_length + window_size * 2) * (window_size * 2 + 1), window_size * 2 + 1, window_size * 2),
                    window_size * 2)
print(vvv)

vvvv = torch.max(vvv, 1)
print(vvvv)
