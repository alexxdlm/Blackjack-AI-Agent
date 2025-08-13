#Register env

from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec


register(
    id="Whitejack-v0",
    entry_point="blackjack_v1:BlackJack_v1",
    kwargs={"sab": True, "natural": False},
)