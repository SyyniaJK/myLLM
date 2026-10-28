from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import numpy as np
import pytest
from edgeinfer.allocation import TPBandwidth
from edgeinfer.diagnostics import (EqualUplink, LengthEstimatePolicy, attenuate, diagnostic_metrics,
    execute_snapshot, snapshot_scenario, _scan_job, replay_g5)
from edgeinfer.experiment import write_json
from edgeinfer.policies import BalancedPolicy
from edgeinfer.scenarios import Scenario
from edgeinfer.snapshot import subset_costs, exact_frontier, plan_cost
from edgeinfer.workload import Request


def test_attenuation_preserves_every_request_and_only_changes_gains(tiny_config):
    source = Scenario.generate(tiny_config, 8)
    changed = attenuate(source, 20)
    assert changed.offers == source.offers and changed.config == source.config
    np.testing.assert_array_equal(changed.up, source.up * 0.01)
    np.testing.assert_array_equal(changed.down, source.down * 0.01)


def test_uplink_ablation_keeps_identical_downlink_solver():
    bits, margins, rates = np.array([100., 200.]), np.array([1., 0.1]), np.array([1000., 2000.])
    np.testing.assert_array_equal(EqualUplink().downlink(bits, margins, rates), TPBandwidth().downlink(bits, margins, rates))
    np.testing.assert_array_equal(EqualUplink().uplink(bits, rates), [0.5, 0.5])


def test_same_batch_full_gpu_charge_invariant_but_upload_readiness_changes():
    scenario = snapshot_scenario(50000, 2, 40)
    action = [1, 1, 2, 1, 2, 2, 0, 0, 0, 0]
    tp = diagnostic_metrics(execute_snapshot(scenario, action, TPBandwidth()))
    eq = diagnostic_metrics(execute_snapshot(scenario, action, EqualUplink()))
    assert tp['gpu_increment_full_j'] == eq['gpu_increment_full_j']
    assert tp['batch_upload_sum_s'] < eq['batch_upload_sum_s']
    assert tp['fixed_j'] == eq['fixed_j']
    assert tp['gpu_increment_full_j'] == pytest.approx(tp['gpu_increment_full_ledger_j'])


@pytest.mark.parametrize('mode', ['cap', 'conditional_mean', 'half_cap'])
def test_causal_length_proxy_does_not_receive_future_outputs(mode):
    source = snapshot_scenario(50000, 4, 0)
    changed = replace(source, offers=tuple(replace(r, output_tokens=1) for r in source.offers))
    from edgeinfer.environment import EdgeEnv
    env = EdgeEnv(source.config)
    obs, info = env.reset(seed=source.seed, options={'scenario': source, 'workload_mode': 'fixed'})
    p0, p1 = LengthEstimatePolicy(source, mode), LengthEstimatePolicy(changed, mode)
    assert p0.truth is None and p1.truth is None
    np.testing.assert_array_equal(p0.act(obs,info['action_mask']),p1.act(obs,info['action_mask']))


def test_optimistic_length_can_create_joint_batch_oom():
    source = snapshot_scenario(50000, 9, 0)
    config = replace(source.config, users=15, device=replace(source.config.device,memory_gb=16))
    offers=tuple(Request(u,0,0.5,512,1024,1024,9) for u in range(12))
    scenario=replace(Scenario.generate(config,50000),offers=offers)
    from edgeinfer.environment import EdgeEnv
    def outcome(mode):
        env=EdgeEnv(config)
        obs,info=env.reset(seed=scenario.seed,options={'scenario':scenario,'workload_mode':'fixed'})
        env.step(LengthEstimatePolicy(scenario,mode).act(obs,info['action_mask']))
        return diagnostic_metrics(env)
    cap, optimistic=outcome('cap'),outcome('half_cap')
    assert cap['oom_requests']==0 and optimistic['oom_requests']>0
    assert cap['goodput_window']>optimistic['goodput_window']


def test_subset_frontier_matches_direct_exhaustive_environment():
    source=snapshot_scenario(50000,2,20)
    scenario=replace(source,offers=(source.offers[0],source.offers[3],source.offers[5]))
    requests,tables=subset_costs(scenario)
    choices,frontier=exact_frontier(tables,len(requests))
    brute={}
    for candidate in product(range(4),repeat=3):
        action=np.zeros(10,dtype=int)
        for r,choice in zip(requests,candidate): action[r.user]=choice
        metric=diagnostic_metrics(execute_snapshot(scenario,action))
        g,e=plan_cost(candidate,tables)
        assert g==metric['goodput_window']
        assert e==pytest.approx(metric['variable_full_j'])
        brute[g]=min(brute.get(g,float('inf')),metric['variable_full_j'])
    assert set(brute)==set(frontier)
    for g in brute: assert frontier[g][0]==pytest.approx(brute[g])
    assert plan_cost(choices,tables)[0]==max(brute)


def test_main_scan_save_and_replay_are_exact(tiny_config,tmp_path,monkeypatch):
    from edgeinfer import diagnostics
    monkeypatch.setattr(diagnostics,'make_policy',lambda method,c,rep,seed: BalancedPolicy(c,seed))
    config=replace(tiny_config,mean_think_s=2.0)
    source=Scenario.generate(config,9)
    source.save(tmp_path/'inputs'/'think_2'/'scenario_9')
    rows=_scan_job((str(tmp_path),2.0,20,'equal_uplink',1,[9]))
    write_json(tmp_path/'scan.json',rows)
    saved=next(r for r in rows if r['method']=='tp_mppo')
    result=replay_g5(tmp_path,saved['case'],'tp_mppo','equal_uplink',1,9,tmp_path/'replay')
    assert result['goodput_window']==saved['goodput_window']
    assert json.loads((tmp_path/'replay'/'replay.json').read_text())['exact']


def test_exported_bundle_supplies_explicit_source_provenance(tmp_path,monkeypatch):
    from edgeinfer.diagnostics import source_commit
    monkeypatch.chdir(tmp_path)
    (tmp_path/'BUNDLE_INFO.json').write_text(json.dumps({'commit':'exported-test-commit'}))
    assert source_commit()=='exported-test-commit'
