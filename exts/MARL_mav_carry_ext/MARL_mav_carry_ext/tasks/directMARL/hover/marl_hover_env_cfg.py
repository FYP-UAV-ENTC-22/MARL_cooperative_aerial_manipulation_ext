# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from MARL_mav_carry_ext.assets import FLYCRANE_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Events for the hovering task.

    Resetting states on resets, disturbances, etc.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 1.5),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["load_link"]),
            "mass_distribution_params": (1.4, 1.4),  # range of mass to randomize,
            "operation": "abs",
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (1.5, 1.5),
    #             "roll": (-0, 0),
    #             "pitch": (-0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #     },
    # )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class MARLHoverEnvCfg(DirectMARLEnvCfg):
    # control mode
    control_mode = "raptor"  # raptor, ACCBR or geometric
    # env
    decimation = 6  # high-level policy at 50 Hz
    episode_length_s = 20

    # history of observations
    partial_obs = True  # if only local observations are used
    history_len = 3

    possible_agents = ["falcon1", "falcon2", "falcon3"]
    num_drones = len(possible_agents)
    if control_mode == "geometric":
        action_dim_geo = 12
        action_spaces = {"falcon1": action_dim_geo, "falcon2": action_dim_geo, "falcon3": action_dim_geo}
        obs_dim_geo = 87
        observation_spaces = {"falcon1": obs_dim_geo, "falcon2": obs_dim_geo, "falcon3": obs_dim_geo}
        state_space = 84
    elif control_mode == "ACCBR":
        action_dim_accbr = 5
        action_spaces = {"falcon1": action_dim_accbr, "falcon2": action_dim_accbr, "falcon3": action_dim_accbr}
        if partial_obs:
            obs_dim_accbr = 45 * history_len
        else:
            obs_dim_accbr = 87
        observation_spaces = {"falcon1": obs_dim_accbr, "falcon2": obs_dim_accbr, "falcon3": obs_dim_accbr}
        state_space = 84
    elif control_mode == "raptor":
        # The policy nudges an absolute position setpoint; the velocity feedforward is the
        # derivative of that same increment, so it needs no outputs of its own.
        action_dim_raptor = 3
        action_spaces = {"falcon1": action_dim_raptor, "falcon2": action_dim_raptor, "falcon3": action_dim_raptor}
        if partial_obs:
            obs_dim_raptor = 45 * history_len
        else:
            obs_dim_raptor = 87
        observation_spaces = {"falcon1": obs_dim_raptor, "falcon2": obs_dim_raptor, "falcon3": obs_dim_raptor}
        state_space = 84

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )
    # robot
    robot_cfg: ArticulationCfg = FLYCRANE_CFG.replace(prim_path="/World/envs/env_.*/flycrane")
    robot_cfg.spawn.activate_contact_sensors = True

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/flycrane/.*", update_period=0.0, history_length=3, debug_vis=False
    )
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")
    contact_sensor_threshold = 0.1

    # falcon CoM names
    falcon_names = "Falcon.*_base_link_inertia"
    # rotor names
    falcon_rotor_names = "Falcon.*_rotor_.*"
    # payload name
    payload_name = "load_odometry_sensor_link"
    # rope name and termination terms
    rope_name = "rope_.*_link"
    # The cable body at the drone end, for a cable made of several bodies. None: the same as rope_name.
    rope_top_name: str | None = None
    cable_angle_limits_drone = 0.0  # cos(angle) limits
    cable_angle_limits_payload = -math.sqrt(2) / 2  # cos(angle) limits
    cable_collision_threshold = 0.2
    cable_collision_num_points = 10
    drone_collision_threshold = 0.6
    bounding_box_threshold = 5.0
    goal_achieved_range = 0.3
    goal_achieved_ori_range = 0.4

    # low level control
    low_level_decimation: int = 3  # RAPTOR at 100 Hz, its native training rate
    max_thrust_pp = 6.25  # N

    # motor model (RAPTOR mode). The defaults are the Falcon's.
    motor_thrust_coeff = 1.562522e-06  # N s^2/rad^2
    motor_torque_coeff = 3.4375484e-08  # N m s^2/rad^2
    motor_time_constant = 0.033  # s
    rotor_directions = (1.0, -1.0, 1.0, -1.0)  # sign of each rotor's yaw torque on the body
    initial_rotor_speed: float | None = None  # rad/s at reset; None = the Falcon's measured hover speeds
    thrust_scale_range = (1.0, 1.0)  # per-drone thrust scale, resampled every episode
    linear_drag = (0.0, 0.0, 0.0)  # N/(m/s) on each drone, body frame

    # Prismatic joints that set each cable's length, on assets that have them. Every episode each cable
    # gets its own offset from the nominal length, drawn from the range.
    cable_length_joint_names: str | None = None
    cable_length_offset_range = (0.0, 0.0)  # m

    # What the policy sees of the payload (the critic always sees the truth).
    payload_obs_delay_steps = 0  # policy steps
    payload_pos_noise_std = 0.0  # m
    payload_rot_noise_std = 0.0  # rad

    # RAPTOR setpoint generation
    setpoint_step_scale = 0.05  # m per high-level tick -> 2.5 m/s of authority at 50 Hz
    # Upper bound on how fast a setpoint may move. The policy's actions are unbounded, so without it
    # setpoint_step_scale is only the speed at unit action. None: no bound.
    setpoint_max_speed: float | None = None  # m/s
    # Anti-windup only: caps how far the setpoint can run ahead of a drone that cannot keep
    # up. It must stay well ABOVE RAPTOR's 0.5 m error clip -- a leash tighter than the clip
    # binds during normal tracking and drags the setpoint along behind a drifting drone,
    # destroying the absolute position reference it exists to provide.
    setpoint_leash = 1.5
    # Rotor speed range the normalised motor command maps onto. Must match RotorMotor.
    raptor_omega_min = 150.0
    raptor_omega_max = 2800.0
    # RAPTOR emits [front-right, back-right, back-left, front-left]. Entry j gives which of
    # those four outputs drives Falcon rotor body j, so the remap is a plain gather. Derive
    # it with
    #   scripts/MARL_mav_carry/testing_scripts/raptor_single_drone.py --check-rotor-order
    raptor_rotor_permutation = (3, 0, 1, 2)  # measured: slots are FL, FR, BR, BL
    # RAPTOR was trained on X-configuration quadrotors, whose arms sit at +-45 deg from the
    # body x axis. If the Falcon is a plus airframe (arms along +-x, +-y) its roll and pitch
    # axes are rotated 45 deg from what RAPTOR assumes, and feeding it the raw body frame
    # makes it pitch when it means to roll. This rotates the frame RAPTOR sees about body z
    # to compensate. Harmless to apply: drone yaw is a free DOF here, since a single cable
    # transmits no yaw torque. Derived by --check-rotor-order; 0.0 for a true X airframe.
    raptor_yaw_offset_deg = 0.0

    # rewards
    pos_track_weight = 1.5
    ori_track_weight = 1.5
    action_smoothness_weight = 0.5  # now penalises setpoint jerk, i.e. trajectory smoothness
    force_penalty_weight = 0.5
    downwash_rew_weight = 0.5

    # goal terms
    goal_range = {
        "pos_x": (-1.0, 1.0),
        "pos_y": (-1.0, 1.0),
        "pos_z": (0.5, 1.5),
        "roll": (-math.pi / 4, math.pi / 4),
        "pitch": (-math.pi / 4, math.pi / 4),
        "yaw": (-math.pi, math.pi),
    }
    range_curriculum_steps = 7500

    make_quat_unique_command = False
    # goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "goal": sim_utils.SphereCfg(
    #             radius=0.0335,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
    #         ),
    #     },
    # )

    # debug visualization
    debug_vis: bool = True
    if debug_vis:
        marker_cfg_goal = FRAME_MARKER_CFG.copy()
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"

        marker_cfg_body = FRAME_MARKER_CFG.copy()
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=True)

    events = EventCfg()
