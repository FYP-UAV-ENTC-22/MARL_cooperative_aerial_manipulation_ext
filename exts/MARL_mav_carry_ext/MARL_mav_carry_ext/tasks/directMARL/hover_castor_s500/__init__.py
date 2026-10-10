# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CASTOR S500 decentralized hover environment: the flycrane hover task on the rig castor-main simulates
(three S500 quadrotors, a disc payload, cables that can go slack), with RAPTOR stepped the way PX4 steps it.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-castor-s500-payload-decentralized-hovering-v0",
    entry_point="MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env:MARLHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_hover_castor_s500_env_cfg:MARLHoverCastorS500EnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
