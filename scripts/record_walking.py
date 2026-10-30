"""Record actual six-step walking or a controlled-stop trial with a foot close-up.

Optional recording dependencies: Pillow and ffmpeg. Headless Linux: MUJOCO_GL=egl.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from humanoid1.walking import WalkTask, WalkProgram, source_fingerprint
from humanoid1.core import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/v07/walking.mp4'); parser.add_argument('--stop-at', type=float)
    args = parser.parse_args(); ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required only for recording')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    sources = source_fingerprint(); model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    task = WalkTask(WalkProgram(stop_at_s=args.stop_at), log=output.with_suffix('.csv'))
    task.r.model.vis.global_.offheight = 640  # Visual buffer only.
    renderer = mujoco.Renderer(task.r.model, height=640, width=640)
    closeup = mujoco.Renderer(task.r.model, height=240, width=384)
    camera = mujoco.MjvCamera(); camera.lookat[:] = [.11, 0., .95]
    camera.distance = 2.8; camera.azimuth = 140; camera.elevation = -12
    foot_camera = mujoco.MjvCamera(); foot_camera.lookat[:] = [.18, 0., .14]
    foot_camera.distance = 1.05; foot_camera.azimuth = 140; foot_camera.elevation = -20
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17); small = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = small = ImageFont.load_default()
    frames = 0; count = 0
    with output.with_suffix('.ffmpeg.log').open('w') as errors:
        process = subprocess.Popen([ffmpeg, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '1024x640', '-r', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '23',
            '-pix_fmt', 'yuv420p', str(output)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            while not task.done:
                task.step(); count += 1
                if count % 2 and not task.done:
                    continue
                renderer.update_scene(task.r.data, camera=camera); closeup.update_scene(task.r.data, camera=foot_camera)
                frame = Image.new('RGB', (1024, 640), (18, 27, 40))
                frame.paste(Image.fromarray(renderer.render()), (0, 0))
                frame.paste(Image.fromarray(closeup.render()), (640, 400))
                draw = ImageDraw.Draw(frame); row = task.last
                title = 'HUMANOID-1 | CONTROLLED STOP' if args.stop_at is not None else 'HUMANOID-1 | SIX-STEP GAIT'
                draw.text((18, 18), title, font=font, fill='white', stroke_width=1, stroke_fill=(18, 27, 40))
                labels = ['Actual MuJoCo simulation', f"Simulation time: {row['time_s']:.2f} s",
                    f"Phase: {row['gait_phase']}", f"Measured completed steps: {row['completed_steps']} / 6",
                    f"Body travel: {row['body_progress_m']*1000:.1f} mm",
                    f"Left sole: {row['left_sole_clearance_m']*1000:+.1f} mm",
                    f"Right sole: {row['right_sole_clearance_m']*1000:+.1f} mm",
                    f"Foot loads: {row['left_normal_n']:.0f} / {row['right_normal_n']:.0f} N",
                    f"Torso tilt: {row['torso_tilt_rad']:.3f} rad",
                    'Stop requested' if row['controlled_stop_requested'] else 'Plan: six steps, then standing',
                    f"Standing dwell: {row['standing_dwell_s']:.2f} s",
                    'Slow, straight, flat-floor gait', 'Joint motors only; no root assistance']
                for i, label in enumerate(labels):
                    draw.text((654, 16+i*27), label, font=small, fill='white')
                draw.text((654, 379), 'FOOT CLOSE-UP | same simulation', font=small, fill=(130, 220, 245))
                process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes()); frames += 1
        finally:
            renderer.close(); closeup.close(); process.stdin.close(); code = process.wait()
    if code:
        raise RuntimeError('ffmpeg failed; see recording log')
    if sources != source_fingerprint() or model_sha != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise RuntimeError('sources changed during recording; rerun for matched evidence')
    result = task.summary()
    result.update(video=str(output), video_frames=frames, video_fps=25,
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), source_sha256=sources, model_sha256=model_sha)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if not result['passed']:
        raise RuntimeError('recorded simulation failed acceptance')
    print(json.dumps({key:result[key] for key in ('passed', 'elapsed_s', 'completed_steps', 'body_progress_m', 'video_frames')}))


if __name__ == '__main__':
    main()
