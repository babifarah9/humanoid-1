import argparse, csv, json
from pathlib import Path
import numpy as np
from .core import Humanoid, StandingPD

def run(seconds=5., log='results/stand.csv', kp=70., kd=10., visualize=False):
    robot=Humanoid(); ctl=StandingPD(robot,kp,kd)
    steps=int(seconds / (robot.model.opt.timestep*10))
    file=Path(log);file.parent.mkdir(parents=True,exist_ok=True)
    viewer=None
    if visualize:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(robot.model,robot.data)
    try:
        with file.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['time_s','pelvis_height_m','root_x_m','root_y_m','stopped','reason']);writer.writeheader()
            for k in range(steps):
                obs,info=robot.step(ctl.act(),frames=10)
                writer.writerow({'time_s':round(robot.data.time,4),'pelvis_height_m':round(robot.data.qpos[2],5),'root_x_m':round(robot.data.qpos[0],5),'root_y_m':round(robot.data.qpos[1],5),'stopped':info['stopped'],'reason':info['reason'] or ''})
                if viewer and viewer.is_running(): viewer.sync()
                if info['stopped']:break
    finally:
        if viewer:viewer.close()
    summary={'elapsed_s':round(robot.data.time,4),'pelvis_height_m':round(float(robot.data.qpos[2]),4),'safety_stop':robot.stopped,'reason':robot.safety_reason(),'actuators':robot.nu,'log':str(file)}
    print(json.dumps(summary,indent=2));return summary

def main():
    p=argparse.ArgumentParser(description='Humanoid-1 local standing test')
    p.add_argument('--seconds',type=float,default=5.);p.add_argument('--log',default='results/stand.csv');p.add_argument('--kp',type=float,default=70.);p.add_argument('--kd',type=float,default=10.);p.add_argument('--view',action='store_true')
    a=p.parse_args();run(a.seconds,a.log,a.kp,a.kd,a.view)
if __name__=='__main__':main()
