#!/bin/bash
# render_chunk.sh K NJOBS TOTAL_FRAMES: render frames [K*T/N, (K+1)*T/N) at 1920x1080 with one worker per CPU,
# then grade + caption + encode them (CRF 15, keyframe every 2 s) and split into <95 MB parts for git.
set -euo pipefail
K=$1; NJ=$2; T=$3; FPS=24
F0=$(( K * T / NJ )); F1=$(( (K + 1) * T / NJ )); P=$(nproc)
S=$(python3 -c "print($F0/$FPS)"); E=$(python3 -c "print($F1/$FPS)")
echo "chunk $K: frames $F0..$((F1 - 1)) with $P workers"
mkdir -p frames out
for w in $(seq 0 $((P - 1))); do
  ( for try in 1 2 3; do node engine/render.js film.html frames 1920x1080 --range "$S" "$E" "$P" "$w" > "frames/_log$w.txt" 2>&1 && break; echo "worker $w retry $try"; done ) &
done
wait
N=$(ls frames/f*.jpg | wc -l); echo "rendered $N of $((F1 - F0))"; grep -h "PAGE ERRORS" -A5 frames/_log*.txt || true
[ "$N" -eq $((F1 - F0)) ] || { echo "MISSING FRAMES"; exit 1; }
python3 engine/finish_film.py frames out/chunk.mp4 --tts tts --size 1920x1080 --start "$F0"
ffmpeg -v error -i out/chunk.mp4 -c copy -f segment -segment_time 6 -reset_timestamps 1 "out/c$(printf %02d $K)_%03d.mp4"
rm out/chunk.mp4; ls -la out
