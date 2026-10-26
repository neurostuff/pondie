"""patch_train.py SRC DST: copy nx/train_prose.py and add a per-optimizer-step line with elapsed time and peak memory
(the original prints peak only in --probe mode). Nothing else changes."""
import sys
s = open(sys.argv[1]).read()
anchor = "            step += 1\n"
assert s.count(anchor) == 1
add = anchor + ('            log(f"  [smoke] step {step}  loss {loss.item():.4f}  t {time.time()-t0:.1f}s  '
                'peak {torch.cuda.max_memory_allocated()/2**20:,.0f} MiB  tok {seen}")\n')
open(sys.argv[2], "w").write(s.replace(anchor, add))
