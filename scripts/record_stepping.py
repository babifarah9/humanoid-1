"""Record real two-step physics, including a close-up of measured foot motion.

Optional recording dependencies: Pillow and ffmpeg. On headless Linux set
MUJOCO_GL=egl. The controller and verification path is identical to the CLI.
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
from humanoid1.stepping import StepTask, StepProgram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/v06/two_steps.mp4')
    args = parser.parse_args()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required only for recording')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    task = StepTask(StepProgram(), log=output.with_suffix('.csv'))
    task.r.model.vis.global_.offheight = 640  # Visual buffer only; physics is unchanged.
    renderer = mujoco.Renderer(task.r.model, height=640, width=640)
    closeup = mujoco.Renderer(task.r.model, height=240, width=384)
    camera = mujoco.MjvCamera(); camera.lookat[:] = [0., 0., .95]
    camera.distance = 2.7; camera.azimuth = 140; camera.elevation = -12
    foot_camera = mujoco.MjvCamera(); foot_camera.lookat[:] = [.08, 0., .15]
    foot_camera.distance = .85; foot_camera.azimuth = 140; foot_camera.elevation = -20
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
        small = ImageFont.truetype('DejaVuSans.ttf', 14)
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
                if count % 2:
                    continue
                renderer.update_scene(task.r.data, camera=camera)
                closeup.update_scene(task.r.data, camera=foot_camera)
                frame = Image.new('RGB', (1024, 640), (18, 27, 40))
                frame.paste(Image.fromarray(renderer.render()), (0, 0))
                frame.paste(Image.fromarray(closeup.render()), (640, 400))
                draw = ImageDraw.Draw(frame); row = task.last
                draw.text((18, 18), 'HUMANOID-1 | TWO-STEP ADVANCE', font=font, fill='white', stroke_width=1, stroke_fill=(18, 27, 40))
                phase = row['planned_support']
                labels = ['Actual MuJoCo simulation', f"Simulation time: {row['time_s']:.2f} s",
                    f'Scheduled support: {phase}', 'Measured sole clearance:',
                    f"  Left  {row['left_sole_clearance_m']*1000:+.1f} mm",
                    f"  Right {row['right_sole_clearance_m']*1000:+.1f} mm",
                    'Measured forward foot travel:',
                    f"  Left  {(row['left_foot_x_m']-task.c.home_feet[0,0])*1000:.1f} mm",
                    f"  Right {(row['right_foot_x_m']-task.c.home_feet[1,0])*1000:.1f} mm",
                    f"Torso tilt: {row['torso_tilt_rad']:.3f} rad",
                    f"Foot loads: {row['left_normal_n']:.0f} / {row['right_normal_n']:.0f} N",
                    'No root support or external assistance', 'Engineered skill; not continuous walking']
                for i, label in enumerate(labels):
                    draw.text((654, 16+i*27), label, font=small, fill='white')
                draw.text((654, 379), 'FOOT CLOSE-UP | same simulation', font=small, fill=(130, 220, 245))
                process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes()); frames += 1
        finally:
            renderer.close(); closeup.close(); process.stdin.close()
            code = process.wait()
    if code:
        raise RuntimeError('ffmpeg failed; see recording log')
    result = task.summary()
    result.update(video=str(output), video_frames=frames, video_fps=25,
                  video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if not result['passed']:
        raise RuntimeError('recorded simulation failed acceptance')
    print(json.dumps({k:result[k] for k in ('passed', 'elapsed_s', 'video_frames', 'video_sha256')}))


if __name__ == '__main__':
    main()
