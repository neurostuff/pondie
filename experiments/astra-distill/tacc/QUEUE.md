# Vista queue (2026-10-23, allocation IBN26027). Code: $WORK/astra-distill/code (local copies in tacc/), configs in code/configs/<run>.json, read at job start.

| job id | run | partition | -t | dependency | notes |
|---|---|---|---|---|---|
| 1095914 | h-smoke (4B train, base eval, merge+serve eval, 27B train) | gh-dev | 1:30 | | done, 12.7 min |
| 1095949 | h-smoke2 (27B tput, 27B merge+serve eval, 4B tput) | gh-dev | 1:15 | | done, 18.1 min |
| 1095970 | h-smoke3 (27B r=64 at 49k, padded throughput) | gh-dev | 1:00 | | done, 9.3 min |
| 1095915 | h34-1k | gh | 1:20 | | NuExtract3, 1,000 papers |
| 1095916 | h34-1k-eval | gh | 2:00 | afterok:1095915 | |
| 1095917 | h34-3k | gh | 3:30 | | |
| 1095918 | h34-3k-eval | gh | 2:00 | afterok:1095917 | |
| 1095919 | h34-10k | gh | 11:00 | | |
| 1095920 | h34-10k-eval | gh | 2:00 | afterok:1095919 | |
| 1095921 | r2-4b-full | gh | 13:30 | | |
| 1095922 | r2-4b-full-eval | gh | 2:00 | afterok:1095921 | |
| 1095923 | r3-27b-a (run r3-27b-full) | gh | 24:00 | | soft-stops 15 min before the limit, exit 75 |
| 1095924 | r3-27b-b (same run, resumes) | gh | 24:00 | afterany:1095923 | |
| 1095972 | r3-27b-c (same run, resumes) | gh | 12:00 | afterany:1095924 | exits at once if DONE |
| 1095973 | r3-27b-full-eval | gh | 6:00 | afterok:1095972 | not run if the epoch is unfinished (exit 75) |
| 1095925 | (cancelled: eval before the third chunk was added) | | | | |
