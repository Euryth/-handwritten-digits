import numpy as np
from src.dropout import Dropout

np.random.seed(0)

dropout = Dropout(drop_prob=0.5)
x = np.ones((100, 100))  # toàn số 1 để dễ kiểm tra tỉ lệ bị tắt

# --- Training mode ---
dropout.train()
out = dropout.forward(x)
zero_ratio = np.mean(out == 0)
print(f"Zero ratio during training (expect ~0.5): {zero_ratio:.3f}")
assert 0.4 < zero_ratio < 0.6, "Drop probability seems incorrect!"

# check scaling: kept values should be scaled to 1/(1-p) = 2.0
kept_values = out[out != 0]
assert np.allclose(kept_values, 2.0), "Inverted dropout scaling is incorrect!"

grad_output = np.ones_like(x)
grad_input = dropout.backward(grad_output)
assert np.array_equal((grad_input != 0), (out != 0)), "Backward mask should match forward mask!"
print("PASS: Dropout training mode works correctly")

# --- Eval mode ---
dropout.eval()
out_eval = dropout.forward(x)
assert np.array_equal(out_eval, x), "Eval mode should not drop anything!"
print("PASS: Dropout eval mode passes through unchanged")