#!/bin/sh
# compare.sh CPU_VMAF VULKAN_VMAF WORKDIR [extra arguments for the Vulkan run]
#
# Scores generated clips (odd sizes, 8 to 16 bit, 4:2:0, 4:2:2, 4:4:4,
# banding) with the VMAF v0.6.1 + NEG and four v1.0.16 models, once with the
# CPU feature extractors and once with the Vulkan ones (--vulkan), and checks
# that every feature of every frame is the same. Needs ffmpeg. For an exact
# comparison, both vmaf tools should print the features with all digits
# (the CI job patches output.c's "%.6f" to "%.17g").
set -e
CPU=$1; VK=$2; W=$3; shift 3
MODELS=$(cd "$(dirname "$0")/../../../model" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W/clips"
rm -rf "$W/cpu" "$W/vulkan"
mkdir -p "$W/cpu" "$W/vulkan"

clip() { # name w h pix_fmt frames source
    [ -f "$W/clips/$1.dis.yuv" ] && return 0
    ffmpeg -nostdin -loglevel error -y -f lavfi -i "$6" -frames:v $5 \
        -vf "scale=$2:$3:flags=bicubic,format=$4" -f rawvideo "$W/clips/$1.ref.yuv"
    ffmpeg -nostdin -loglevel error -y -f rawvideo -pix_fmt $4 -s $2x$3 -r 24 \
        -i "$W/clips/$1.ref.yuv" -vf "boxblur=1:1,noise=alls=8:all_seed=7" \
        -f rawvideo -pix_fmt $4 "$W/clips/$1.dis.yuv"
}
REAL="mandelbrot=s=1920x1080:r=24,noise=alls=12:allf=t"
TS="testsrc2=s=1920x1080:r=24"
BAND="gradients=s=1920x1080:r=24:speed=0.02,format=yuv420p"
CLIPS="odd8:621:349:yuv420p:420:8:$REAL
odd10:621:349:yuv420p10le:420:10:$TS
hd10:1920:1080:yuv420p10le:420:10:$REAL
band8:1280:720:yuv420p:420:8:$BAND
s444:640:360:yuv444p:444:8:$TS
s422:641:361:yuv422p12le:422:12:$REAL
s16:640:360:yuv444p16le:444:16:$REAL"

echo "$CLIPS" | while IFS=: read n w h pf p b src; do
    clip $n $w $h $pf 6 "$src"
    for run in "$CPU:$W/cpu:" "$VK:$W/vulkan:--vulkan"; do
        bin=${run%%:*}; rest=${run#*:}; out=${rest%%:*}; vk=${rest#*:}
        "$bin" -r "$W/clips/$n.ref.yuv" -d "$W/clips/$n.dis.yuv" -w $w -h $h -p $p -b $b \
            --json -o "$out/$n.v0.json" -q $vk ${vk:+"$@"} \
            -m path="$MODELS/vmaf_v0.6.1.json":name=v061 \
            -m path="$MODELS/vmaf_v0.6.1neg.json":name=neg
        "$bin" -r "$W/clips/$n.ref.yuv" -d "$W/clips/$n.dis.yuv" -w $w -h $h -p $p -b $b \
            --json -o "$out/$n.v1.json" -q $vk ${vk:+"$@"} \
            -m path="$MODELS/vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json":name=a \
            -m path="$MODELS/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json":name=b \
            -m path="$MODELS/vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json":name=c \
            -m path="$MODELS/vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json":name=d
    done
done
python3 "$HERE/compare.py" "$W/cpu" "$W/vulkan"
