# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Flyhex decentralized hover environment — six MAVs, two rows of three anchors.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-flycrane-payload-decentralized-hovering-flyhex-v0",
    entry_point=f"{__name__}.marl_hover_flyhex_env:MARLHoverFlyhexEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_hover_flyhex_env_cfg:MARLHoverFlyhexEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
