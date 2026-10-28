from dataclasses import replace
import numpy as np
import pytest
from edgeinfer.config import Config, Device, Model
from edgeinfer.environment import EdgeEnv


@pytest.fixture
def tiny_config():
    return replace(Config(), users=2, nodes=2, slots=6, radius_m=10,
                   input_tokens=(2,), output_caps=(2,), output_fractions=(1.0,), slo_s=(10.0,),
                   model=Model("tiny_test", 100, 2, 1, 4), device=Device("test", 0.01, 1.0))


@pytest.fixture
def injected_env(tiny_config):
    def build(requests, *, config=tiny_config, energy=True, allocator=None):
        env = EdgeEnv(config, energy=energy, allocator=allocator)
        env.reset(seed=123)
        env.current_requests = {r.user: r for r in requests}
        env.workload.next_arrival[:] = np.inf
        return env
    return build

