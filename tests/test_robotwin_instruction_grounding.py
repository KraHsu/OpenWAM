"""The collector's safe metadata branch cannot fall through to expert motion."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.robotwin_constraint_policy_candidates import ground_instruction_info


@pytest.mark.parametrize('task',['blocks_ranking_size','place_bread_basket'])
def test_audited_metadata_bypasses_expert_and_preserves_scene(task):
    native=SimpleNamespace(scene='initial')
    info={'{A}':'large block'} if task=='blocks_ranking_size' else {'{A}':'076_breadbasket/base0','{B}':'075_bread/base1','{a}':'left'}
    def forbidden(*a,**kw):raise AssertionError('Safe initialization attempted expert motion')
    native.get_info=native.move=native.pick_and_place_block=native.remove=forbidden
    reader=lambda task,env:dict(info=deepcopy(info),provenance={'expert_actions_for_instruction':False})
    actual,provenance=ground_instruction_info(task,native,readonly_reader=reader,expert=forbidden,
                                             fingerprint=lambda env:env.scene,expected_scene='initial')
    assert actual==info and native.scene=='initial'
    assert provenance['scene_before']==provenance['scene_after']=='initial'


def test_scene_change_or_wrong_initial_scene_cannot_be_silently_accepted():
    native=SimpleNamespace(scene='initial')
    def reader(task,env):
        env.scene='moved';return {'info':{},'provenance':{}}
    def forbidden(*a):raise AssertionError('Must not execute expert')
    with pytest.raises(ValueError,match='changed the scene'):
        ground_instruction_info('blocks_ranking_size',native,readonly_reader=reader,expert=forbidden,
                                fingerprint=lambda env:env.scene,expected_scene='initial')
    with pytest.raises(ValueError,match='differs from frozen'):
        ground_instruction_info('blocks_ranking_size',native,readonly_reader=forbidden,expert=forbidden,
                                fingerprint=lambda env:env.scene,expected_scene='initial')


def test_unsupported_task_retains_explicit_isolated_expert_fallback():
    native=SimpleNamespace(scene='initial',plan_success=True,check_success=lambda:True)
    called=[]
    def expert(env):called.append(env);return {'info':{'{a}':'dual'}}
    info,provenance=ground_instruction_info('lift_pot',native,readonly_reader=lambda *args:None,expert=expert,
                                           fingerprint=lambda env:env.scene,expected_scene='initial')
    assert info=={'{a}':'dual'} and called==[native]
    assert provenance['adapter']=='isolated_expert_then_scene_reset'
