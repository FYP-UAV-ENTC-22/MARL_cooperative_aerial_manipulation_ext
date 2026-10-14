# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import numpy as np
import torch
from collections.abc import Sequence

from MARL_mav_carry_ext.controllers import GeometricController, IndiController, RaptorController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
from MARL_mav_carry_ext.controllers.raptor import action_to_omega, assemble_observation
from MARL_mav_carry_ext.planning import SetpointIntegrator, policy_frame
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import get_drone_pdist, get_drone_rpos
from MARL_mav_carry_ext.vehicle import FlightStack, raptor_core_observation

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer, DelayBuffer
from isaaclab.utils.math import (
    compute_pose_error,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_apply,
    quat_apply_inverse,
    quat_unique,
    sample_uniform,
)

from .marl_hover_env_cfg import MARLHoverEnvCfg


class MARLHoverEnv(DirectMARLEnv):
    cfg: MARLHoverEnvCfg

    def __init__(self, cfg: MARLHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # body indices
        self._falcon_idx = self.robot.find_bodies(cfg.falcon_names)[0]
        self._falcon_rotor_idx = self.robot.find_bodies(cfg.falcon_rotor_names)[0]
        self._payload_idx = self.robot.find_bodies(cfg.payload_name)[0]
        self._rope_idx = self.robot.find_bodies(cfg.rope_name)[0]
        # The cable body at the drone end. A rigid rod is one body, so both ends are the same.
        self._rope_top_idx = self.robot.find_bodies(cfg.rope_top_name or cfg.rope_name)[0]

        # configuration
        self._num_drones = len(self._falcon_idx)
        self._control_mode = cfg.control_mode

        # # observation buffers
        self._observation_buffers = {}
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent] = CircularBuffer(cfg.history_len, self.num_envs, device=self.device)

        # action buffers
        # buffers
        self._forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)
        # self.setpoint_delay_buffers = {}
        # for agent in self.cfg.possible_agents:
        #     self.setpoint_delay_buffers[agent] = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        #     self.setpoint_delay_buffers[agent].set_time_lag(cfg.constant_delay)
        self._setpoints = {}
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 12, device=self.device)
            elif self._control_mode == "ACCBR":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 5, device=self.device)
            elif self._control_mode == "raptor":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 3, device=self.device)

        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)
        self.drone_orientations[..., 0] = 1.0
        self.drone_rot_matrices = torch.zeros(self.num_envs, self._num_drones, 3, 3, device=self.device)
        self.drone_linear_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_linear_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

        # outer loop controller
        self.geo_controllers = {}
        self._indi_controllers = {}
        self.raptor = None
        if self._control_mode == "raptor":
            # A single frozen policy replaces both the geometric and the INDI loop, and is
            # evaluated for every drone of every environment in one batched call.
            self.raptor = RaptorController(self.num_envs, self._num_drones, self.device)
        else:
            for i in range(self._num_drones):
                self.geo_controllers[i] = GeometricController(self.num_envs, self._control_mode)
                self._indi_controllers[i] = IndiController(self.num_envs)
        self._ll_counter = 0
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)

        # The position setpoint each drone's RAPTOR is given: absolute, seeded from the drone on every reset, moved
        # by the high-level policy (planning/setpoint.py).
        self.setpoint = SetpointIntegrator(
            self.num_envs,
            self._num_drones,
            self.step_dt,
            cfg.setpoint_step_scale,
            cfg.setpoint_leash,
            cfg.setpoint_max_speed,
            self.device,
        )
        # What sits between the simulated bodies and the two controllers on the aircraft: PX4's estimator, the DDS
        # and Wi-Fi links, the ESCs (vehicle/flight_stack.py). None: both controllers are given the true state.
        self.stack: FlightStack | None = None
        if cfg.flight_stack is not None:
            if self._control_mode != "raptor" or cfg.low_level_decimation != 1 or cfg.raptor_native_every is None:
                raise ValueError("flight_stack needs RAPTOR called on every physics step, as PX4 calls it")
            self.stack = FlightStack(
                cfg.flight_stack, self.num_envs, self._num_drones, self.sim.get_physics_dt(), self.device
            )
        self._stack_reset_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # the drones as the policy was last given them: what its setpoints are seeded from and leashed to
        self._policy_drone_positions = self.drone_positions
        self._policy_drone_yaw = torch.zeros(self.num_envs, self._num_drones, device=self.device)
        self._rotor_permutation = torch.tensor(cfg.raptor_rotor_permutation, device=self.device)
        # Rotation from the Falcon body frame into the virtual X-configuration frame RAPTOR
        # was trained on. Identity when the airframe is already X.
        theta = math.radians(cfg.raptor_yaw_offset_deg)
        self._raptor_frame_rotation = torch.tensor(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        # motor model
        self.motor_models = {}
        if cfg.initial_rotor_speed is None:
            # the Falcon's hover speeds, experimentally obtained
            initial_rpms = [
                torch.tensor([[1441.5819, 1351.1626, 1341.0111, 1428.5597]], device=self.device).repeat(self.num_envs, 1),
                torch.tensor([[1377.9199, 1451.8428, 1408.9022, 1329.2014]], device=self.device).repeat(self.num_envs, 1),
                torch.tensor([[1281.3964, 1293.0708, 1361.7539, 1347.2434]], device=self.device).repeat(self.num_envs, 1),
            ]
        else:
            initial_rpms = [
                torch.full((self.num_envs, 4), cfg.initial_rotor_speed, device=self.device)
                for _ in range(self._num_drones)
            ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(
                self.num_envs,
                initial_rpms[i],
                thrust_coeff=cfg.motor_thrust_coeff,
                torque_coeff=cfg.motor_torque_coeff,
                tau_up=cfg.motor_time_constant,
                tau_down=cfg.motor_time_constant,
                directions=cfg.rotor_directions,
            )
        self._drag_coefficients = torch.tensor(cfg.linear_drag, device=self.device)
        self._drag_forces = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._cable_length_joint_ids = None
        if cfg.cable_length_joint_names is not None:
            self._cable_length_joint_ids = self.robot.find_joints(cfg.cable_length_joint_names)[0]
        self._cable_slide_joint_ids = None
        if cfg.cable_slide_joint_names is not None:
            self._cable_slide_joint_ids = self.robot.find_joints(cfg.cable_slide_joint_names)[0]
        self._cable_slack_time = torch.zeros(self.num_envs, self._num_drones, device=self.device)
        self._cable_tension = torch.zeros(self.num_envs, self._num_drones, device=self.device)
        self.cable_slack = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.drone_overspeed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._drone_speed = torch.zeros(self.num_envs, self._num_drones, device=self.device)
        # each agent's observation frame can arrive late, by a number of policy steps fixed for the episode
        self._frame_delays = None
        if cfg.obs_delay_steps_range[1] > 0:
            self._frame_delays = {
                agent: DelayBuffer(cfg.obs_delay_steps_range[1], self.num_envs, device=self.device)
                for agent in cfg.possible_agents
            }
        if cfg.motor_time_constant_range is not None:
            self._motor_tau = torch.full((self.num_envs, 1), cfg.motor_time_constant, device=self.device)
            for motor_model in self.motor_models.values():
                motor_model.tau_up = motor_model.tau_down = self._motor_tau
        # Per-environment thrust scale, resampled on reset: a worn propeller, a sagging battery.
        self._thrust_scale = torch.ones(self.num_envs, self._num_drones, 1, device=self.device)
        # What the policy sees of the payload can lag and carry noise; the critic sees the truth.
        self._payload_obs_delay = None
        if cfg.payload_obs_delay_steps > 0:
            self._payload_obs_delay = DelayBuffer(cfg.payload_obs_delay_steps, self.num_envs, device=self.device)
            self._payload_obs_delay.set_time_lag(cfg.payload_obs_delay_steps)
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # load buffers
        self.load_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_orientation = torch.zeros(self.num_envs, 4, device=self.device)
        self.load_orientation[:, 0] = 1.0
        self.current_load_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.load_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_length_x = torch.tensor([[0.275, 0, 0]] * self.num_envs, device=self.device)
        self.load_length_y = torch.tensor([[0, 0.275, 0]] * self.num_envs, device=self.device)

        # Goal terms
        # # goal buffers
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0

        self.goal_pos_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.difference_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device)

        self.goal_dist_counter = torch.zeros(self.num_envs, device=self.device)

        # reward logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_reward",
                "ori_reward",
                "action_smoothness",
                "force_penalty",
                "downwash_reward",
                "slack_penalty",
                "hold_still",
            ]
        }

        # # -- metrics
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # termination buffers
        self.falcon_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.payload_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.illegal_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.angle_limit_drone = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.angle_limit_load = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.cable_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.drone_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.body_pos_outside = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # debug vis
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        contact_sensors = ContactSensor(self.cfg.contact_forces)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)  # TODO: not sure what this does
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_forces"] = contact_sensors
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self.cfg.possible_agents:
            # terms used for smoothness reward, current and previously calculated actions
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

            # introduce delay in the setpoints
            # actions[agent][:] = self.setpoint_delay_buffers[agent].compute(actions[agent])

        if self._control_mode == "raptor":
            self._update_setpoints(actions)
            return

        for drone, action in actions.items():

            if self._control_mode == "geometric":
                self._setpoints[drone]["pos"] = action[:, :3]
                self._setpoints[drone]["lin_vel"] = action[:, 3:6]
                self._setpoints[drone]["lin_acc"] = action[:, 6:9]
                self._setpoints[drone]["jerk"] = action[:, 9:12]

                # self._desired_position[:, i] = self.drone_setpoint[i]["pos"]

            elif self._control_mode == "ACCBR":
                self._setpoints[drone]["lin_acc"] = action[:, :3]
                self._setpoints[drone]["body_rates"] = torch.cat((action[:, 3:], self._constant_yaw), dim=-1)

            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = self._constant_yaw
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    # the setpoint state under the names it has always had
    @property
    def sp_pos(self) -> torch.Tensor:
        return self.setpoint.position

    @property
    def sp_vel(self) -> torch.Tensor:
        return self.setpoint.velocity

    @property
    def sp_yaw(self) -> torch.Tensor:
        return self.setpoint.yaw

    @property
    def _sp_seed_pending(self) -> torch.Tensor:
        return self.setpoint.seed_pending

    def _update_setpoints(self, actions: dict[str, torch.Tensor]) -> None:
        """Advance the per-drone position setpoint from the high-level actions (planning/setpoint.py).

        Seeded from, and leashed to, the drone as the policy knows it: the true pose without a flight stack, the
        odometry that has reached the companion computer with one.
        """
        if self.stack is None:
            # drone_positions was refreshed by the observation pass that follows every reset
            position = self.drone_positions
            yaw = torch.atan2(self.drone_rot_matrices[..., 1, 0], self.drone_rot_matrices[..., 0, 0])
        else:
            position, yaw = self._policy_drone_positions, self._policy_drone_yaw
        self.setpoint.step(torch.stack([actions[agent] for agent in self.cfg.possible_agents], dim=1), position, yaw)
        if self.stack is not None:
            self.stack.command(self.setpoint.position, self.setpoint.velocity, self.setpoint.yaw)

    def _apply_action(self) -> None:
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            if self._control_mode == "raptor":
                self._compute_raptor_thrusts()
                self._ll_counter = 0
                self._apply_rotor_wrench()
                self._ll_counter += 1
                return

            all_thrusts = []
            all_moments = []

            drone_positions = self.robot.data.body_com_state_w[
                :, self._falcon_idx, :3
            ] - self.scene.env_origins.unsqueeze(1)
            drone_orientations = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
            drone_linear_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
            drone_angular_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]
            drone_linear_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, :3]
            drone_angular_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, 3:6]

            self.drone_positions[:] = drone_positions  # + torch.randn_like(drone_positions) * self.position_noise_std
            self.drone_orientations[:] = (
                drone_orientations  # + torch.randn_like(drone_orientations) * self.orientation_noise_std
            )
            self.drone_linear_velocities[:] = (
                drone_linear_velocities  # + torch.randn_like(drone_linear_velocities) * self.linear_velocity_noise_std
            )
            self.drone_angular_velocities[:] = (
                drone_angular_velocities  # + torch.randn_like(drone_angular_velocities) * self.angular_velocity_noise_std
            )
            self.drone_linear_accelerations[:] = (
                drone_linear_accelerations  # + torch.randn_like(drone_linear_accelerations) * self.linear_acceleration_noise_std
            )
            self.drone_angular_accelerations[:] = (
                drone_angular_accelerations  # + torch.randn_like(drone_angular_accelerations) * self.angular_acceleration_noise_std
            )

            for i in range(self._num_drones):
                drone_states: dict = {}  # dict of tensors
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                # calculate current jerk
                self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (self.step_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"]
                )

                # alpha_cmd, acc_load, acc_cmd, q_cmd, target_rpm = self.geo_controllers[i].getCommand(
                #     drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"]
                # )

                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load
                )

                # target_rpm = self._indi_controllers[i].getCommand(
                #     drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"])

                # if self.cfg.debug_vis:
                #     self.drone_positions_debug[:, i] = drone_states["pos"] + self._env.scene.env_origins
                #     if self._control_mode == "geometric":
                #         self.drone_goals_debug[:, i] = self.drone_setpoint[i]["pos"] + self._env.scene.env_origins
                #     self.des_acc_debug[:, i] = acc_cmd
                #     self.des_ori_debug[:, i] = q_cmd

                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(target_rpm, self.sampling_time)
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            forces = torch.cat(all_thrusts, dim=-1)
            torques = torch.cat(all_moments, dim=-1)
            self._forces[..., 2] = forces
            self._moments[..., 2] = torques.view(self.num_envs, self._num_drones, 4).sum(-1)
            self._ll_counter = 0
        self._ll_counter += 1

        self._apply_rotor_wrench()

    def _compute_raptor_thrusts(self) -> None:
        """Run the RAPTOR inner loop for every drone and turn its output into rotor forces.

        Mirrors the observation assembly the flight controller performs in
        ``mc_raptor.cpp:344``. The sign flips there are PX4's NED/FRD -> FLU conversion and
        are deliberately absent here: Isaac is already FLU.
        """
        self._read_drone_state()
        # The simulator reports angular velocity in the world frame; RAPTOR expects it in the body frame, as the
        # INDI controller also assumes (indi.py:101).
        angular_velocity_body = quat_apply_inverse(self.drone_orientations, self.drone_angular_velocities)

        if self.stack is None:
            inputs = (
                self.drone_positions,
                self.drone_rot_matrices,
                self.drone_linear_velocities,
                angular_velocity_body,
                self.sp_pos,
                self.sp_vel,
                self.sp_yaw,
            )
        else:
            # PX4's estimate of the drone, and the setpoint as it has arrived over the link
            payload = self.robot.data.body_com_state_w[:, self._payload_idx].squeeze(1)
            self.stack.step(
                self.drone_positions,
                self.drone_rot_matrices,
                self.drone_linear_velocities,
                angular_velocity_body,
                payload[:, :3] - self.scene.env_origins,
                matrix_from_quat(payload[:, 3:7]),
            )
            inputs = self.stack.raptor_inputs()

        # Errors and attitude in the target-yaw frame, as mc_raptor.cpp:352 does; for an airframe that is not an X,
        # attitude and body rates in the virtual X-configuration frame (vehicle/raptor_mode.py).
        position_error, orientation, velocity_error, angular_velocity_body = raptor_core_observation(
            *inputs, frame_rotation=self._raptor_frame_rotation if self.cfg.raptor_yaw_offset_deg != 0.0 else None
        )

        # Linear drag acts in the body frame and is applied there, together with the rotor torques.
        self._drag_forces[:] = -self._drag_coefficients * quat_apply_inverse(
            self.drone_orientations, self.drone_linear_velocities
        )

        observation = assemble_observation(
            position_error,
            orientation,
            velocity_error,
            angular_velocity_body,
            self.raptor.previous_action,
        )
        if self.cfg.raptor_native_every is None:
            action = self.raptor(observation)
        else:
            # the executor supplies the previous-action input itself
            action = self.raptor.step_px4(observation[..., : -self.raptor.ACTION_DIM], self.cfg.raptor_native_every)

        # RAPTOR emits [front-right, back-right, back-left, front-left]; reorder into the
        # order the rotor bodies were found in before handing over to the motor model.
        target_omega = action_to_omega(
            action[..., self._rotor_permutation], self.cfg.raptor_omega_min, self.cfg.raptor_omega_max
        )
        if self.stack is not None:
            target_omega = self.stack.esc(target_omega)

        all_thrusts = []
        all_moments = []
        for i in range(self._num_drones):
            thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(
                target_omega[:, i], self.sampling_time
            )
            all_thrusts.append(thrusts)
            all_moments.append(moments)

        thrusts = torch.cat(all_thrusts, dim=-1).view(self.num_envs, self._num_drones, 4) * self._thrust_scale
        moments = torch.cat(all_moments, dim=-1).view(self.num_envs, self._num_drones, 4) * self._thrust_scale
        self._forces[..., 2] = thrusts.view(self.num_envs, -1)
        self._moments[..., 2] = moments.sum(-1)

    def _read_drone_state(self) -> None:
        """Refresh the true state of every drone, at the point the task calls the drone's position."""
        data = self.robot.data
        state = (data.body_link_state_w if self.cfg.drone_point == "origin" else data.body_com_state_w)[
            :, self._falcon_idx
        ]
        self.drone_positions[:] = state[..., :3] - self.scene.env_origins.unsqueeze(1)
        self.drone_orientations[:] = state[..., 3:7]
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)
        self.drone_linear_velocities[:] = state[..., 7:10]
        self.drone_angular_velocities[:] = state[..., 10:13]

    def _apply_rotor_wrench(self) -> None:
        """Push the current rotor forces and reaction torques into the articulation."""
        # apply torques induced by rotors to each body
        self.robot.set_external_force_and_torque(self._drag_forces, self._moments, body_ids=self._falcon_idx)
        # apply forces to each rotor
        self.robot.set_external_force_and_torque(self._forces, torch.zeros_like(self._forces), body_ids=self._falcon_rotor_idx)

    def _get_observations(self) -> dict[str, torch.Tensor]:

        # local observations include:
        # load state
        # ego-drone drone
        # other-drone_states
        # goal terms

        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )
        # _get_dones refreshes this too, but before any reset: without the read here the first
        # observation of an episode carries the previous episode's payload orientation.
        self.load_orientation[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 3:7].squeeze(1)
        self.current_load_matrix[:] = matrix_from_quat(self.load_orientation)
        self.load_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 7:10].squeeze(1)
        self.load_ang_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 10:13].squeeze(1)

        self._read_drone_state()

        self.goal_pos_error[:] = self.pose_command_w[:, :3] - self.load_position
        goal_load_matrix = matrix_from_quat(self.pose_command_w[:, 3:7])
        self.difference_matrix[:] = torch.matmul(goal_load_matrix, self.current_load_matrix.transpose(1, 2))

        # action_histories = []
        # for agent in self.cfg.possible_agents:
        #     if self.setpoint_delay_buffers[agent]._circular_buffer._buffer is None:
        #         action_history = self.actions[agent].unsqueeze(1).repeat(1, self.cfg.max_delay + 1, 1)
        #         action_histories.append(action_history)
        #     else:
        #         action_histories.append(self.setpoint_delay_buffers[agent]._circular_buffer.buffer)
        # self.all_action_histories = torch.cat(action_histories, dim=-1)

        if self.cfg.partial_obs:
            return self._partial_observations(goal_load_matrix)
        else:
            obs_falcon1 = torch.cat(
                (
                    self.load_position,
                    self.current_load_matrix.view(self.num_envs, -1),
                    self.load_vel,
                    self.load_ang_vel,
                    # drone terms
                    torch.tensor([[1, 0, 0]] * self.num_envs, device=self.device),  # one-hot encoding
                    self.drone_positions.view(self.num_envs, -1),
                    self.drone_rot_matrices.view(self.num_envs, -1),
                    self.drone_linear_velocities.view(self.num_envs, -1),
                    self.drone_angular_velocities.view(self.num_envs, -1),
                    self.goal_pos_error,
                    self.difference_matrix.view(self.num_envs, -1),
                    # self.all_action_histories.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

            obs_falcon2 = torch.cat(
                (
                    self.load_position,
                    self.current_load_matrix.view(self.num_envs, -1),
                    self.load_vel,
                    self.load_ang_vel,
                    # drone terms
                    torch.tensor([[0, 1, 0]] * self.num_envs, device=self.device),  # one-hot encoding
                    self.drone_positions.view(self.num_envs, -1),
                    self.drone_rot_matrices.view(self.num_envs, -1),
                    self.drone_linear_velocities.view(self.num_envs, -1),
                    self.drone_angular_velocities.view(self.num_envs, -1),
                    self.goal_pos_error,
                    self.difference_matrix.view(self.num_envs, -1),
                    # self.all_action_histories.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

            obs_falcon3 = torch.cat(
                (
                    self.load_position,
                    self.current_load_matrix.view(self.num_envs, -1),
                    self.load_vel,
                    self.load_ang_vel,
                    # drone terms
                    torch.tensor([[0, 0, 1]] * self.num_envs, device=self.device),  # one-hot encoding
                    self.drone_positions.view(self.num_envs, -1),
                    self.drone_rot_matrices.view(self.num_envs, -1),
                    self.drone_linear_velocities.view(self.num_envs, -1),
                    self.drone_angular_velocities.view(self.num_envs, -1),
                    self.goal_pos_error,
                    self.difference_matrix.view(self.num_envs, -1),
                    # self.all_action_histories.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

        observations = {
            "falcon1": obs_falcon1,
            "falcon2": obs_falcon2,
            "falcon3": obs_falcon3,
        }
        return observations

    def _partial_observations(self, goal_load_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each agent's own view: the payload, its own drone, the goal error; `history_len` frames of it.

        Without a flight stack every agent is given the same payload pose and the true state (with whatever noise
        and delay the task configures). With one, each agent is given what has reached its companion computer: its
        drone's odometry over DDS and the payload's pose over its own Wi-Fi link.
        """
        if self.stack is None:
            load_position, load_matrix, goal_pos_error, difference_matrix = self._payload_observation(goal_load_matrix)
            drone_positions, drone_rot_matrices, drone_linear_velocities, drone_angular_velocities = (
                self._drone_observation()
            )
            payload = [(load_position, load_matrix, goal_pos_error, difference_matrix)] * self._num_drones
        else:
            if self._stack_reset_pending.any():
                self._reset_stack(self._stack_reset_pending.nonzero(as_tuple=False).squeeze(-1))
            drone_positions, drone_rot_matrices, drone_linear_velocities, drone_angular_velocities = (
                self.stack.policy_drone_state()
            )
            self._policy_drone_positions = drone_positions
            self._policy_drone_yaw = torch.atan2(drone_rot_matrices[..., 1, 0], drone_rot_matrices[..., 0, 0])
            load_position, load_matrix = self.stack.policy_payload_pose()
            goal_pos_error = self.pose_command_w[:, None, :3] - load_position
            difference_matrix = goal_load_matrix.unsqueeze(1) @ load_matrix.transpose(-1, -2)
            payload = [
                (load_position[:, i], load_matrix[:, i], goal_pos_error[:, i], difference_matrix[:, i])
                for i in range(self._num_drones)
            ]

        observations = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            load_position_i, load_matrix_i, goal_pos_error_i, difference_matrix_i = payload[i]
            frame = policy_frame(
                i,
                self._num_drones,
                load_position_i,
                load_matrix_i,
                drone_positions[:, i],
                drone_rot_matrices[:, i],
                drone_linear_velocities[:, i],
                drone_angular_velocities[:, i],
                goal_pos_error_i,
                difference_matrix_i,
            )
            if self._frame_delays is not None:
                frame = self._frame_delays[agent].compute(frame)
            self._observation_buffers[agent].append(frame)
            observations[agent] = self._observation_buffers[agent].buffer.reshape(self.num_envs, -1)
        return observations

    def _reset_stack(self, env_ids: torch.Tensor) -> None:
        """Start the flight stack of freshly reset environments on their (now readable) spawn state."""
        self.stack.reset(
            env_ids,
            self.drone_positions,
            self.drone_rot_matrices,
            self.drone_linear_velocities,
            quat_apply_inverse(self.drone_orientations, self.drone_angular_velocities),
            self.load_position,
            self.current_load_matrix,
            torch.stack([self.motor_models[i].current_omega for i in range(self._num_drones)], dim=1),
        )
        self._stack_reset_pending[env_ids] = False

    def _drone_observation(self):
        """Each drone's state as its own policy sees it: the estimator's output, so with noise."""
        cfg = self.cfg
        position, rotation = self.drone_positions, self.drone_rot_matrices
        lin_vel, ang_vel = self.drone_linear_velocities, self.drone_angular_velocities
        if cfg.drone_pos_noise_std > 0.0:
            position = position + torch.randn_like(position) * cfg.drone_pos_noise_std
        if cfg.drone_rot_noise_std > 0.0:
            axis = torch.nn.functional.normalize(torch.randn_like(position), dim=-1).view(-1, 3)
            angle = torch.randn(axis.shape[0], device=self.device) * cfg.drone_rot_noise_std
            noisy = quat_mul(quat_from_angle_axis(angle, axis), self.drone_orientations.view(-1, 4))
            rotation = matrix_from_quat(noisy).view(self.num_envs, self._num_drones, 3, 3)
        if cfg.drone_lin_vel_noise_std > 0.0:
            lin_vel = lin_vel + torch.randn_like(lin_vel) * cfg.drone_lin_vel_noise_std
        if cfg.drone_ang_vel_noise_std > 0.0:
            ang_vel = ang_vel + torch.randn_like(ang_vel) * cfg.drone_ang_vel_noise_std
        return position, rotation, lin_vel, ang_vel

    def _payload_observation(self, goal_load_matrix: torch.Tensor):
        """The payload pose as the policy sees it, and the goal errors computed from it.

        On the aircraft the payload pose is an estimate that reaches each drone over a radio link, so
        it can be delayed by ``payload_obs_delay_steps`` policy steps and carry noise. With both off
        this returns the true values.
        """
        cfg = self.cfg
        if self._payload_obs_delay is None and cfg.payload_pos_noise_std == 0.0 and cfg.payload_rot_noise_std == 0.0:
            return self.load_position, self.current_load_matrix, self.goal_pos_error, self.difference_matrix

        pose = torch.cat((self.load_position, self.load_orientation), dim=-1)
        if self._payload_obs_delay is not None:
            pose = self._payload_obs_delay.compute(pose)
        position, orientation = pose[:, :3], pose[:, 3:7]
        if cfg.payload_pos_noise_std > 0.0:
            position = position + torch.randn_like(position) * cfg.payload_pos_noise_std
        if cfg.payload_rot_noise_std > 0.0:
            # a small rotation about a random axis
            axis = torch.nn.functional.normalize(torch.randn_like(position), dim=-1)
            angle = torch.randn(self.num_envs, device=self.device) * cfg.payload_rot_noise_std
            orientation = quat_mul(quat_from_angle_axis(angle, axis), orientation)
        load_matrix = matrix_from_quat(orientation)
        goal_pos_error = self.pose_command_w[:, :3] - position
        difference_matrix = torch.matmul(goal_load_matrix, load_matrix.transpose(1, 2))
        return position, load_matrix, goal_pos_error, difference_matrix

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # load terms
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # drone terms
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                # goal terms
                self.goal_pos_error,
                self.difference_matrix.view(self.num_envs, -1),
                # self.all_action_histories.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )

        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # pos reward
        goal_pos_error_norm = torch.norm(self.pose_command_w[:, :3] - self.load_position, dim=-1)
        reward_distance_scale = 1.5
        reward_position = (
            self.cfg.pos_track_weight * torch.exp(-goal_pos_error_norm * reward_distance_scale) * self.step_dt
        )

        # orientation reward
        orientation_error = quat_error_magnitude(self.pose_command_w[:, 3:7], self.load_orientation)
        reward_distance_scale = 1.5
        reward_orientation = (
            self.cfg.ori_track_weight * torch.exp(-orientation_error * reward_distance_scale) * self.step_dt
        )

        # action smoothness reward
        current_actions = torch.cat([self.actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        action_prev = torch.cat([self.prev_actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        diff_action = ((current_actions - action_prev).abs()) / self._num_drones
        reward_action_smoothness = (
            self.cfg.action_smoothness_weight * torch.exp(-torch.norm(diff_action, dim=-1).square()) * self.step_dt
        )

        # force penalty
        normalized_forces = self._forces[..., 2] / self.cfg.max_thrust_pp
        effort_sum = torch.max(normalized_forces, dim=-1)[0]
        reward_effort = self.cfg.force_penalty_weight * torch.exp(-effort_sum) * self.step_dt

        # downwash reward
        reward_downwash = self.cfg.downwash_rew_weight * self._downwash_reward() * self.step_dt

        # cables pulling less than the reference tension
        reward_slack = torch.zeros_like(reward_position)
        if self._cable_slide_joint_ids is not None and self.cfg.slack_penalty_weight > 0.0:
            low_tension = (1.0 - self._cable_tension / self.cfg.cable_tension_ref).clamp(0.0, 1.0)
            reward_slack = -self.cfg.slack_penalty_weight * low_tension.mean(dim=-1) * self.step_dt

        # drones flying faster than the soft speed limit
        if self.cfg.drone_speed_limit is not None and self.cfg.drone_speed_penalty_weight > 0.0:
            span = max(self.cfg.drone_speed_limit - self.cfg.drone_speed_soft_limit, 1e-6)
            over = ((self._drone_speed - self.cfg.drone_speed_soft_limit) / span).clamp(0.0, 1.0)
            reward_slack = reward_slack - self.cfg.drone_speed_penalty_weight * over.mean(dim=-1) * self.step_dt

        # drones at rest once the payload is at its goal (true drone speed, as the speed limit above)
        reward_hold_still = torch.zeros_like(reward_position)
        if self.cfg.hold_still_weight > 0.0:
            at_goal = torch.exp(-((goal_pos_error_norm / self.cfg.hold_still_goal_gate) ** 2))
            still = torch.exp(-self._drone_speed.mean(dim=-1) / self.cfg.hold_still_speed_scale)
            reward_hold_still = self.cfg.hold_still_weight * at_goal * still * self.step_dt

        rewards = {
            "hold_still": reward_hold_still,
            "slack_penalty": reward_slack,
            "pos_reward": reward_position,
            "ori_reward": reward_orientation,
            "action_smoothness": reward_action_smoothness,
            "force_penalty": reward_effort,
            "downwash_reward": reward_downwash,
        }

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        shared_rewards = (
            reward_position
            + reward_orientation
            + reward_action_smoothness
            + reward_effort
            + reward_downwash
            + reward_slack
            + reward_hold_still
        )

        return {agent: shared_rewards for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The terminations for the environment. Since all of the agents are connected by the cables,
        if 1 agent terminates, terminate all agents.
        """
        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )

        # crashing into ground
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)
        self.payload_fly_low = self.load_position[:, 2] < 0.1

        # illegal contact
        contact_sensor = self.scene.sensors[self.cfg.sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        # check if any contact force exceeds the threshold
        self.illegal_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self.cfg.sensor_cfg.body_ids], dim=-1), dim=1)[0]
            > self.cfg.contact_sensor_threshold,
            dim=1,
        )

        # angle limits
        rope_orientations_world = self.robot.data.body_com_state_w[:, self._rope_idx, 3:7].view(-1, 4)
        rope_top_orientations_world = self.robot.data.body_com_state_w[:, self._rope_top_idx, 3:7].view(-1, 4)
        drone_orientation_world = self.drone_orientations.view(-1, 4)
        drone_orientation_inv = quat_inv(drone_orientation_world)
        rope_orientations_drones = quat_mul(
            drone_orientation_inv, rope_top_orientations_world
        )  # cable angles relative to drones
        roll_drone, pitch_drone, _ = euler_xyz_from_quat(rope_orientations_drones)  # yaw can be whatever
        mapped_angle_drone = torch.stack((torch.cos(roll_drone), torch.cos(pitch_drone)), dim=1)
        self.angle_limit_drone = (
            (mapped_angle_drone < self.cfg.cable_angle_limits_drone).any(dim=1).view(-1, self._num_drones).any(dim=1)
        )

        self.load_orientation[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 3:7].squeeze(1)
        payload_orientation_world = self.load_orientation.repeat(1, self._num_drones, 1).view(-1, 4)
        payload_orientation_inv = quat_inv(payload_orientation_world)
        rope_orientations_payload = quat_mul(
            payload_orientation_inv, rope_orientations_world
        )  # cable angles relative to payload
        roll_load, pitch_load, _ = euler_xyz_from_quat(rope_orientations_payload)  # yaw can be whatever
        mapped_angle_load = torch.stack((torch.cos(roll_load), torch.cos(pitch_load)), dim=1)
        self.angle_limit_load = (
            (mapped_angle_load < self.cfg.cable_angle_limits_payload).any(dim=1).view(-1, self._num_drones).any(dim=1)
        )

        # cables colliding
        self.cable_collision = self._cable_collision(
            self.cfg.cable_collision_threshold, self.cfg.cable_collision_num_points
        )

        # drones colliding
        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = (
            pdist.min(dim=-1).values.min(dim=-1).values
        )  # get the smallest distance between drones in the swarm
        self.drone_collision = separation < self.cfg.drone_collision_threshold

        # bounding box
        self.body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)

        # update metrics
        self._update_metrics()

        # A cable that stays slack: the payload is not being carried. A brief dip during a fast move is fine.
        if self._cable_slide_joint_ids is not None:
            shortfall = -self.robot.data.joint_pos[:, self._cable_slide_joint_ids]
            slack = shortfall > self.cfg.cable_slack_tolerance
            self._cable_slack_time = torch.where(slack, self._cable_slack_time + self.step_dt, 0.0)
            self.cable_slack = (self._cable_slack_time > self.cfg.cable_slack_time).any(dim=-1)
            # The slide joint only pushes back on its limit, so its reaction along the cable is the tension.
            wrench = self.robot.data.body_incoming_joint_wrench_b[:, self._rope_top_idx]
            self._cable_tension = wrench[..., 2].abs()

        drone_speed = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10].norm(dim=-1)
        self._drone_speed = drone_speed
        if self.cfg.drone_speed_limit is not None:
            self.drone_overspeed = (drone_speed > self.cfg.drone_speed_limit).any(dim=-1)

        if self.cfg.max_body_speed is not None:
            speed = self.robot.data.body_com_state_w[..., 7:10].norm(dim=-1).max(dim=-1).values
            self.body_pos_outside = self.body_pos_outside | ~(speed < self.cfg.max_body_speed)  # also catches NaN

        # reset when episode ends
        terminations = (
            self.cable_slack
            | self.drone_overspeed
            | self.falcon_fly_low
            | self.payload_fly_low
            | self.illegal_contact
            | self.angle_limit_drone
            | self.angle_limit_load
            | self.cable_collision
            | self.drone_collision
            | self.body_pos_outside
        )
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        timed_outs = self.time_out

        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)
        self._reset_target_pose(env_ids)

        if self._cable_length_joint_ids is not None:
            low, high = self.cfg.cable_length_offset_range
            offsets = sample_uniform(
                low, high, (len(env_ids), len(self._cable_length_joint_ids)), device=self.device
            )
            self.robot.write_joint_state_to_sim(
                offsets, torch.zeros_like(offsets), joint_ids=self._cable_length_joint_ids, env_ids=env_ids
            )
            self.robot.set_joint_position_target(offsets, joint_ids=self._cable_length_joint_ids, env_ids=env_ids)

        if self._control_mode == "raptor":
            # Defer seeding the setpoint until the spawn pose has been read back, and clear
            # the policy's recurrent state: the GRU carries RAPTOR's online estimate of the
            # vehicle it is flying, which is stale for a drone that just teleported.
            self._stack_reset_pending[env_ids] = True
            if self.cfg.raptor_native_every is None:
                self.raptor.reset(env_ids)
            else:
                self.raptor.reset_px4(env_ids)

            steps = self.setpoint.steps[env_ids].clamp(min=1.0)
            self.extras.setdefault("log", dict())["Metrics/setpoint/leash_active_fraction"] = (
                self.setpoint.leash_active_steps[env_ids] / steps
            ).mean()
            self.extras["log"]["Metrics/setpoint/tracking_error"] = (
                (self.sp_pos[env_ids] - self.drone_positions[env_ids]).norm(dim=-1).mean()
            )
            # Defer seeding the setpoint until the spawn pose has been read back.
            self.setpoint.reset(env_ids)
            # Rotor speeds persist through the first-order lag, so they have to come back to
            # the spawn value too -- otherwise the policy starts from a reset history against
            # motors still spinning at whatever the previous episode ended on.
            for motor_model in self.motor_models.values():
                motor_model.reset(env_ids)
            if self.cfg.motor_time_constant_range is not None:
                low, high = self.cfg.motor_time_constant_range
                self._motor_tau[env_ids] = sample_uniform(low, high, (len(env_ids), 1), device=self.device)
            low, high = self.cfg.thrust_scale_range
            self._thrust_scale[env_ids] = sample_uniform(
                low, high, (len(env_ids), self._num_drones, 1), device=self.device
            )

        for agent in self.cfg.possible_agents:
            # self.setpoint_delay_buffers[agent].reset(env_ids)
            self._observation_buffers[agent].reset(env_ids)
        if self._payload_obs_delay is not None:
            self._payload_obs_delay.reset(env_ids)
        if self._frame_delays is not None:
            low, high = self.cfg.obs_delay_steps_range
            lags = torch.randint(low, high + 1, (len(env_ids),), device=self.device, dtype=torch.int)
            for delay in self._frame_delays.values():
                delay.reset(env_ids)
                delay.set_time_lag(lags, env_ids)
        self._cable_slack_time[env_ids] = 0.0

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        if self.cfg.drone_speed_limit is not None:
            self.extras["log"]["Episode_Termination/drone_overspeed"] = torch.count_nonzero(
                self.drone_overspeed[env_ids]
            ).item()
            self.extras["log"]["Metrics/drone_speed_max"] = self._drone_speed.max(dim=-1).values.mean()
        if self._cable_slide_joint_ids is not None:
            self.extras["log"]["Episode_Termination/cable_slack"] = torch.count_nonzero(self.cable_slack[env_ids]).item()
            self.extras["log"]["Metrics/cable/tension_mean"] = self._cable_tension.mean()
            self.extras["log"]["Metrics/cable/tension_min"] = self._cable_tension.min(dim=-1).values.mean()
        self.extras["log"]["Episode_Termination/angle_drones_cable"] = torch.count_nonzero(
            self.angle_limit_drone[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/angle_load_cable"] = torch.count_nonzero(
            self.angle_limit_load[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/cables_collide"] = torch.count_nonzero(
            self.cable_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/drones_collide"] = torch.count_nonzero(
            self.drone_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            self.body_pos_outside[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/payload_fly_low"] = torch.count_nonzero(
            self.payload_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/illegal_contact"] = torch.count_nonzero(
            self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.time_out[env_ids]).item()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # log metrics
        for metric_name, metric_value in self.metrics.items():
            self.extras["log"][f"Metrics/pose_command/{metric_name}"] = metric_value.mean()

        # reset the action history
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        # if self.common_step_counter > self.cfg.range_curriculum_steps:
        #     self.cfg.goal_range ={
        #     "pos_x": (-2.0, 2.0),
        #     "pos_y": (-2.0, 2.0),
        #     "pos_z": (0.5, 2.5),
        #     "roll": (-math.pi/4, math.pi/4),
        #     "pitch": (-math.pi/4, math.pi/4),
        #     "yaw": (-math.pi, math.pi),
        # }

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.goal_range["pos_x"])
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.goal_range["pos_y"])
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.goal_range["pos_z"])
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.goal_range["roll"])
        euler_angles[:, 1].uniform_(*self.cfg.goal_range["pitch"])
        euler_angles[:, 2].uniform_(*self.cfg.goal_range["yaw"])
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique_command else quat

    def _update_metrics(self):
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.load_position,
            self.load_orientation,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _cable_collision(
        self,
        threshold: float = 0.0,
        num_points: int = 5,
    ) -> torch.Tensor:
        """Check for collisions between cables.

        A collision is detected if the minimum Euclidean distance between any two points
        on different cables is below the threshold.
        """
        cable_bottom_pos_env = self.robot.data.body_com_state_w[
            :, self._rope_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        cable_directions = self.drone_positions - cable_bottom_pos_env  # (num_envs, num_cables, 3)

        # Create linearly spaced points for interpolation (num_points,)
        linspace_points = torch.linspace(0, 1, num_points, device=self.device).view(
            1, 1, num_points, 1
        )  # (1, 1, num_points, 1)

        # Compute cable points (num_envs, num_cables, num_points, 3)
        cable_points = cable_bottom_pos_env.unsqueeze(2) + linspace_points * cable_directions.unsqueeze(
            2
        )  # (num_envs, num_cables, num_points, 3)

        # Flatten cable points for easier distance calculation (num_envs, num_cables * num_points, 3)
        cable_points_flat = cable_points.view(self.num_envs, -1, 3)

        # Pairwise distance calculation
        cable_points_a = cable_points_flat.unsqueeze(2)  # (num_envs, num_points_total, 1, 3)
        cable_points_b = cable_points_flat.unsqueeze(1)  # (num_envs, 1, num_points_total, 3)
        pairwise_diff = cable_points_a - cable_points_b  # (num_envs, num_points_total, num_points_total, 3)
        pairwise_distances = torch.norm(pairwise_diff, dim=-1)  # (num_envs, num_points_total, num_points_total)

        # Mask to ignore self-distances and distances within the same cable
        num_cables = cable_bottom_pos_env.shape[1]
        points_per_cable = num_points

        # Create mask to ignore points on the same cable
        cable_indices = torch.arange(num_cables, device=self.device).repeat_interleave(
            points_per_cable
        )  # (num_points_total,)
        same_cable_mask = cable_indices.unsqueeze(0) == cable_indices.unsqueeze(
            1
        )  # (num_points_total, num_points_total)
        same_cable_mask = same_cable_mask.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )  # (num_envs, num_points_total, num_points_total)

        # Apply mask: set ignored distances to a large value
        pairwise_distances[same_cable_mask] = 1000.0

        # Find the minimum distance across all points in each environment
        min_distances, _ = torch.min(pairwise_distances.view(self.num_envs, -1), dim=-1)  # Shape: (num_envs,)

        # Check if the minimum distance is below the threshold
        is_cable_collision = min_distances < threshold  # Shape: (num_envs,)

        assert is_cable_collision.shape == (self.num_envs,)
        return is_cable_collision

    def _downwash_reward(self):
        # Plane equation for the payload
        x_len_payload_env = quat_apply(self.load_orientation, self.load_length_x)
        y_len_payload_env = quat_apply(self.load_orientation, self.load_length_y)
        edge_payload_x = self.load_position + x_len_payload_env
        edge_payload_y = self.load_position + y_len_payload_env
        plane_vec1 = edge_payload_x - self.load_position
        plane_vec2 = edge_payload_y - self.load_position
        normal = torch.linalg.cross(plane_vec1, plane_vec2)
        d = torch.sum(normal * self.load_position, dim=-1).unsqueeze(-1).unsqueeze(-1)  # Shape (num_envs, 1, 1)

        # Line equations for each drone's thrust direction
        thrust_directions = quat_apply(
            self.drone_orientations.view(-1, 4),
            torch.tensor([[0, 0, 1.0]] * self.num_envs * self._num_drones, device=self.device),
        ).view(self.num_envs, self._num_drones, 3)

        # Calculate intersection points with the plane
        # t = (d - normal * drone_pos) / (normal * thrust_direction)
        numerator = d - torch.sum(normal.unsqueeze(1) * self.drone_positions, dim=-1, keepdim=True)
        denominator = (
            torch.sum(normal.unsqueeze(1) * thrust_directions, dim=-1, keepdim=True) + 1e-6
        )  # Avoid division by zero
        t = numerator / denominator

        # Intersection points on the plane for each drone
        line_point_proj = self.drone_positions + t * thrust_directions  # Shape (num_envs, num_drones, 3)

        # Calculate distance between intersection points and payload position
        line_dist = torch.norm(
            line_point_proj - self.load_position.unsqueeze(1), dim=-1
        )  # Shape (num_envs, num_drones)
        # Reward: penalize based on distance from the intersection point to the payload position
        scaling_factor = 3
        reward_downwash = 1 - torch.exp(
            -torch.min(line_dist, dim=-1).values * scaling_factor
        )  # Min distance from the payload
        return reward_downwash

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not hasattr(self, "goal_pose_visualizer"):
            # -- goal pose
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_goal)
            # -- current body pose
            self.body_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_body)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3] + self.scene.env_origins, self.pose_command_w[:, 3:]
        )

        # -- current body pose
        self.body_pose_visualizer.visualize(self.load_position + self.scene.env_origins, self.load_orientation)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
