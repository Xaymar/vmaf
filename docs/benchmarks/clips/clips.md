# Benchmark clips

Source: `cabin-in-the-woods_3840x2160@60_yuv444-TV-sRGB.mp4`, start offset 20 s (`-ss 20`), 120 frames.
FFmpeg: `ffmpeg version 2026-11-16-git-d35e13b9e0-full_build-www.gyan.dev`

Raw 8-bit 4:2:0 `ref_*`/`dis_*.yuv` are not committed; hashes in `SHA256SUMS.txt`. `dis_1080.mkv` and `dis_4k.mkv` are the x264 files they were decoded from.

Commands (from `make_frames8.sh`, looped over `4k` with `-vf format=yuv420p` and `1080` with `-vf scale=1920:1080:flags=lanczos,format=yuv420p`; `$s` = 3840x2160 or 1920x1080):

```sh
ffmpeg -hide_banner -loglevel error -y -ss 20 -i "$src" -frames:v 120 -vf "$vf" -f rawvideo -pix_fmt yuv420p ref_$n.yuv
ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt yuv420p -s $s -r 60 -i ref_$n.yuv -c:v libx264 -preset medium -crf 30 dis_$n.mkv
ffmpeg -hide_banner -loglevel error -y -i dis_$n.mkv -f rawvideo -pix_fmt yuv420p dis_$n.yuv
```
