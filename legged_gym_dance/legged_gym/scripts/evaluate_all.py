import os
import sys

if __name__ == "__main__":
    task_name = sys.argv[1]
    folder_name = sys.argv[2]
    ckpt_start = int(sys.argv[3])
    ckpt_end = int(sys.argv[4])
    run_name = os.path.basename(folder_name)
    assert run_name != ""
    folder_dir = os.path.dirname(folder_name)
    for ckpt in range(ckpt_start, ckpt_end + 1):
        if os.path.exists(os.path.join(folder_name, "model_%d.pt" % ckpt)):
            os.system("python scripts/play.py --task=%s --load_run %s --checkpoint %d --headless" % (task_name, run_name, ckpt))
            os.system(f"ffmpeg -r 50 -i {folder_dir}/exported/frames/%d.png -pix_fmt yuv420p {folder_name}/output_{ckpt}.mp4")