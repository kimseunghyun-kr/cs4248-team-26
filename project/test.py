import os
import torch
import torch.nn.functional as F

CACHES = ["cache/roberta", "cache/bert"]
PAIRS = ["question-mark", "repeated-Q-marks", "exclamation-mark",
         "very-short", "internet-laughter", "emoticon"]

for cache in CACHES:
    d2 = os.path.join(cache, "conditions", "d2_cbdc")
    try:
        a = torch.load(os.path.join(d2, "bias_anchors.pt"), map_location="cpu").float()
        p = torch.load(os.path.join(d2, "class_prompt_prototypes.pt"), map_location="cpu").float()
    except FileNotFoundError as e:
        print(f"\n{cache}: missing -> {e}\n")
        continue

    cos = (F.normalize(a, dim=-1) @ F.normalize(p, dim=-1).T).numpy()
    print(f"\n{cache}  (anchors={cos.shape[0]}, prototypes={cos.shape[1]})")
    print(f"{'pair':<20}{'neg':>9}{'neu':>9}{'pos':>9}{'|max|':>9}")
    print("-" * 56)
    for i in range(cos.shape[0]):
        neg, neu, pos = cos[i, 0], cos[i, 1], cos[i, 2]
        mx = max(abs(neg), abs(neu), abs(pos))
        name = PAIRS[i] if i < len(PAIRS) else f"anchor{i}"
        print(f"{name:<20}{neg:>9.3f}{neu:>9.3f}{pos:>9.3f}{mx:>9.3f}")
