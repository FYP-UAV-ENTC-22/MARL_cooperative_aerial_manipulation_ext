# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CASTOR S500 decentralized hover environment with the flight stack of the aircraft between the simulated rig and the
controllers: PX4's estimator feeding RAPTOR and, over the DDS link, the policy; the payload's own estimator and its
Wi-Fi link; the ESCs.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-castor-s500-px4-payload-decentralized-hovering-v0",
    entry_point="MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env:MARLHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_hover_castor_s500_px4_env_cfg:MARLHoverCastorS500Px4EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
