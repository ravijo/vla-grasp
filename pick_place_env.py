import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

INSTRUCTION = "pick up the bottle and place it on the tray"
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Scene constants
TABLE_CENTER = np.array([0.5, 0.0])
TABLE_SURFACE_Z = 0.625
PANDA_EE_INDEX = 11
PANDA_ARM_JOINTS = list(range(7))
PANDA_FINGER_JOINTS = [9, 10]
FINGER_OPEN = 0.04
START_JOINTS = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.78]
TRAY_SCALE = 0.30
BOTTLE_H = 0.12  # bottle cylinder height
BOTTLE_R = 0.032  # bottle cylinder radius

# Randomization workspace
WORKSPACE_HALF = 0.15
MIN_OBJ_SEPARATION = 0.16

MAX_ACTION_STEP = 0.03  # clip per-step EE delta (m)
CONTROL_SUBSTEPS = 12  # sim steps per env.step (~20 Hz control)


class PickPlaceEnv:
    def __init__(self, gui=False, seed=0):
        self.rng = np.random.RandomState(seed)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0.63],
            useFixedBase=True,
        )
        for i in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(
                self.robot,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
            )

        self.obj = None
        self.tray = None
        self.ee_down_quat = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

        # Fixed third-person camera
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.15, 0.0, 1.05],
            cameraTargetPosition=[0.5, 0.0, 0.65],
            cameraUpVector=[0, 0, 1],
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=55,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.0,
        )

    def _reset_arm(self):
        for i, a in enumerate(START_JOINTS):
            p.resetJointState(self.robot, i, a)
            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                a,
                force=200,
            )
        for f in PANDA_FINGER_JOINTS:
            p.resetJointState(self.robot, f, FINGER_OPEN)
            p.setJointMotorControl2(
                self.robot,
                f,
                p.POSITION_CONTROL,
                FINGER_OPEN,
                force=30,
            )

    def _sample_positions(self):
        while True:
            obj_xy = TABLE_CENTER + \
                self.rng.uniform(-WORKSPACE_HALF, WORKSPACE_HALF, size=2)
            tray_xy = TABLE_CENTER + \
                self.rng.uniform(-WORKSPACE_HALF, WORKSPACE_HALF, size=2)
            if np.linalg.norm(obj_xy - tray_xy) >= MIN_OBJ_SEPARATION:
                return obj_xy, tray_xy

    def ee_pos(self):
        return np.array(p.getLinkState(self.robot, PANDA_EE_INDEX)[4])

    def _finger_width(self):
        return np.mean([p.getJointState(self.robot, f)[0] for f in PANDA_FINGER_JOINTS])

    def reset(self):
        if self.obj is not None:
            p.removeBody(self.obj)
            p.removeBody(self.tray)
        self._reset_arm()

        obj_xy, tray_xy = self._sample_positions()
        self.tray = p.loadURDF(
            "tray/tray.urdf",
            basePosition=[*tray_xy, TABLE_SURFACE_Z],
            globalScaling=TRAY_SCALE,
        )
        self.obj = p.loadURDF(
            os.path.join(ASSETS, "bottle.urdf"),
            basePosition=[*obj_xy, TABLE_SURFACE_Z + BOTTLE_H / 2 + 0.005],
        )
        p.changeDynamics(self.obj, -1, lateralFriction=2.0)
        for f in PANDA_FINGER_JOINTS:
            p.changeDynamics(self.robot, f, lateralFriction=2.0)

        for _ in range(120):
            p.stepSimulation()
        return self._obs()

    def _obs(self):
        return {
            "image": self.render(),
            "ee_pos": self.ee_pos(),
            "obj_pos": np.array(p.getBasePositionAndOrientation(self.obj)[0]),
            "tray_pos": np.array(p.getBasePositionAndOrientation(self.tray)[0]),
            "finger_width": self._finger_width()
        }

    def step(self, action):
        action = np.asarray(action, dtype=float)
        delta = np.clip(action[:3], -MAX_ACTION_STEP, MAX_ACTION_STEP)
        gripper_cmd = action[6]
        target = self.ee_pos() + delta
        target_finger = FINGER_OPEN * max(0.0, (gripper_cmd + 1.0) / 2.0)

        ik = p.calculateInverseKinematics(
            self.robot,
            PANDA_EE_INDEX,
            target.tolist(),
            self.ee_down_quat,
            maxNumIterations=80,
            residualThreshold=1e-4,
        )
        for _ in range(CONTROL_SUBSTEPS):
            for i in PANDA_ARM_JOINTS:
                p.setJointMotorControl2(
                    self.robot,
                    i,
                    p.POSITION_CONTROL,
                    ik[i],
                    force=200,
                )
            for f in PANDA_FINGER_JOINTS:
                # Modest grip force which is enough to hold the 0.1 kg bottle,
                # low enough to avoid crushing into it.
                p.setJointMotorControl2(
                    self.robot,
                    f,
                    p.POSITION_CONTROL,
                    target_finger,
                    force=20,
                )
            p.stepSimulation()

        obs = self._obs()
        success = self._success(obs)
        return obs, float(success), success, {}

    def _success(self, obs):
        obj, tray = obs["obj_pos"], obs["tray_pos"]
        on_xy = np.linalg.norm(obj[:2] - tray[:2]) < 0.08
        resting = (TABLE_SURFACE_Z + 0.005) < obj[2] < (TABLE_SURFACE_Z + 0.14)
        # Gripper must be OPEN (finger width > 90% of fully open) to count as a successful release.
        released = obs["finger_width"] > 0.9 * FINGER_OPEN
        return bool(on_xy and resting and released)

    def render(self):
        _, _, rgb, _, _ = p.getCameraImage(
            224, 224,
            self.view_matrix,
            self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        p.disconnect(self.cid)


class ExpertPolicy:
    """Scripted state machine using privileged state. Emits 7-DoF EE-delta actions."""

    # Grasp near the TOP of the bottle so the palm (link 8) stays above the
    # bottle top (grasping at mid-body plunged the whole hand into it).
    GRASP_Z = TABLE_SURFACE_Z + BOTTLE_H - 0.03  # fingertips ~3 cm below the cap
    PRE_Z = TABLE_SURFACE_Z + 0.26
    RELEASE_Z = TABLE_SURFACE_Z + 0.15  # drop height above tray walls

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 0
        self.grip = +1.0
        self.timer = 0

    @staticmethod
    def _go(cur, tgt, gain=1.5):
        d = (np.asarray(tgt) - np.asarray(cur)) * gain
        return np.clip(d, -MAX_ACTION_STEP, MAX_ACTION_STEP)

    def act(self, obs):
        ee, obj, tray = obs["ee_pos"], obs["obj_pos"], obs["tray_pos"]
        a = np.zeros(7, dtype=float)

        if self.phase == 0:  # align above bottle at approach height
            tgt = [obj[0], obj[1], self.PRE_Z]
            a[:3] = self._go(ee, tgt)
            if np.linalg.norm(ee[:2] - obj[:2]) < 0.01 and abs(ee[2] - self.PRE_Z) < 0.03:
                self.phase = 1
        elif self.phase == 1:  # descend onto bottle body
            tgt = [obj[0], obj[1], self.GRASP_Z]
            a[:3] = self._go(ee, tgt)
            if ee[2] < self.GRASP_Z + 0.015:
                self.phase = 2
                self.timer = 0
        elif self.phase == 2:  # close gripper
            self.grip = -1.0
            self.timer += 1
            if self.timer > 8:
                self.phase = 3
        elif self.phase == 3:  # lift
            self.grip = -1.0
            a[:3] = self._go(ee, [obj[0], obj[1], self.PRE_Z])
            if ee[2] > self.PRE_Z - 0.02:
                self.phase = 4
        elif self.phase == 4:  # move above tray
            self.grip = -1.0
            a[:3] = self._go(ee, [tray[0], tray[1], self.PRE_Z])
            if np.linalg.norm(ee[:2] - tray[:2]) < 0.02:
                self.phase = 5
        elif self.phase == 5:  # descend over tray
            self.grip = -1.0
            a[:3] = self._go(ee, [tray[0], tray[1], self.RELEASE_Z])
            if ee[2] < self.RELEASE_Z + 0.02:
                self.phase = 6
                self.timer = 0
        elif self.phase == 6:  # release
            self.grip = +1.0
            self.timer += 1
        a[6] = self.grip
        return a

    @property
    def done(self):
        return self.phase == 6 and self.timer > 8


def smoke_test(n_episodes=8, gui=False, max_steps=200):
    env = PickPlaceEnv(gui=gui, seed=0)
    expert = ExpertPolicy()
    successes = 0
    for ep in range(n_episodes):
        obs = env.reset()
        expert.reset()
        succ = False
        for t in range(max_steps):
            obs, r, done, _ = env.step(expert.act(obs))
            if gui:
                time.sleep(1.0 / 120.0)
            if done:
                succ = True
                break
            if expert.done:
                succ = env._success(obs)
                break
        successes += int(succ)
        print(f"  episode {ep}: success={succ} (steps={t + 1})")
    print(
        f"Expert success rate: {successes}/{n_episodes} = {100 * successes / n_episodes:.0f}%")
    env.close()


if __name__ == "__main__":
    smoke_test(n_episodes=8, gui=False)
