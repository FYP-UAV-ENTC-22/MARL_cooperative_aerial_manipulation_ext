# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The flycrane hover task with the CASTOR payload and cable length.

Drones, rigid-rod cables, RAPTOR inner loop, observations, actions, rewards and terminations are the
flycrane hover task's, unchanged. The payload (a 0.4 kg disc with three anchors 120 degrees apart on its
rim) and the cable length come from ``assets/data/AMR/flycastor_data/castor_rig.yaml``.
"""

from MARL_mav_carry_ext.assets import FLYCASTOR_CFG
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import EventCfg, MARLHoverEnvCfg

import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class CastorEventCfg(EventCfg):
    """Resets as in the flycrane task; the payload keeps the mass castor_rig.yaml gives it."""

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["load_link"]),
            "mass_distribution_params": (0.4, 0.4),
            "operation": "abs",
        },
    )


@configclass
class MARLHoverCastorEnvCfg(MARLHoverEnvCfg):
    # robot: same prim path as the flycrane, so the contact sensor and body names carry over
    robot_cfg: ArticulationCfg = FLYCASTOR_CFG.replace(prim_path="/World/envs/env_.*/flycrane")
    robot_cfg.spawn.activate_contact_sensors = True

    events = CastorEventCfg()
