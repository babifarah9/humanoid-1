"""Record actual MuJoCo simulation with measured COM/load overlays.

Optional media dependencies: Pillow and the ffmpeg executable. For a headless
Linux render use MUJOCO_GL=egl. The physics/controller path is unchanged.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from humanoid1.weight_shift import ShiftTask,ShiftProgram


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/v05/weight_shift.mp4')
    a=p.parse_args()
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg is required only for recording')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    task=ShiftTask(ShiftProgram(),log=output.with_suffix('.csv'))
    renderer=mujoco.Renderer(task.r.model,height=480,width=640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,1.];camera.distance=2.6;camera.azimuth=150;camera.elevation=-15
    try:font=ImageFont.truetype('DejaVuSans.ttf',18);small=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=small=ImageFont.load_default()
    log=output.with_suffix('.ffmpeg.log').open('w')
    proc=subprocess.Popen([ffmpeg,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
        '-s','1024x480','-r','25','-i','pipe:0','-an','-c:v','libx264','-crf','24','-pix_fmt','yuv420p',str(output)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    count=0;frames=0
    try:
        while not task.done:
            task.step();count+=1
            if count%2:continue
            renderer.update_scene(task.r.data,camera=camera)
            image=Image.new('RGB',(1024,480),(18,27,40));image.paste(Image.fromarray(renderer.render()),(0,0))
            draw=ImageDraw.Draw(image);row=task.last
            labels=['HUMANOID-1 WEIGHT SHIFT','Actual MuJoCo simulation',f"Simulation time: {row['time_s']:.2f} s",'16 actuators | 32.8 kg',
                f"Command COM y: {row['command_y_m']*1000:+.2f} mm",f"Actual COM y: {row['com_y']*1000:+.2f} mm",
                f"Tracking error: {row['tracking_xy_m']*1000:.3f} mm",f"Torso tilt: {row['torso_tilt_rad']:.5f} rad",'Measured vertical foot loads']
            for i,text in enumerate(labels):draw.text((660,20+i*32),text,font=font if i==0 else small,fill='white')
            fraction=row['left_load_fraction'];x=660;y=330;width=340
            draw.rectangle((x,y,x+width*fraction,y+28),fill=(67,159,218))
            draw.rectangle((x+width*fraction,y,x+width,y+28),fill=(239,156,67))
            draw.text((660,370),f"Left {100*fraction:.1f}% | Right {100*(1-fraction):.1f}%",font=font,fill='white')
            draw.text((660,410),'Engineered control; no learned policy',font=small,fill=(190,210,230))
            draw.text((660,438),'Double support; this is not walking',font=small,fill=(190,210,230))
            proc.stdin.write(np.asarray(image,dtype=np.uint8).tobytes());frames+=1
    finally:
        renderer.close();proc.stdin.close();code=proc.wait();log.close()
    if code:raise RuntimeError('ffmpeg failed; see '+str(output.with_suffix('.ffmpeg.log')))
    result=task.summary();result.update(video_frames=frames,video_fps=25,video=str(output))
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('recorded simulation failed acceptance')
    print(json.dumps({k:result[k] for k in ('elapsed_s','passed','video_frames','video_fps','video')}))


if __name__=='__main__':main()
