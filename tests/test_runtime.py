import numpy as np
from humanoid1.core import Humanoid, StandingPD

def test_simulation():
    robot=Humanoid()
    assert robot.nu==16
    controller=StandingPD(robot)
    for _ in range(10):
        obs,info=robot.step(controller.act())
        assert np.isfinite(obs).all()
    print('PASS: 10 MuJoCo control steps executed; stopped:',robot.stopped)

if __name__=='__main__':test_simulation()
