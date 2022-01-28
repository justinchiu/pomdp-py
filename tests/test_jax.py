import copy
import time
import random
import math

import pomdp_py
import pytest

import jax
import jax.numpy as np

# imports copy-pasted from pomcp
from pomdp_py.framework.basics import Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel
from pomdp_py.framework.planner import Planner
from pomdp_py.representations.distribution.particles import Particles
from pomdp_py.representations.belief.particles import particle_reinvigoration
from pomdp_py.algorithms.po_uct import VNode, RootVNode, QNode, POUCT, RandomRollout
from pomdp_py.algorithms.pomcp_jax import PomcpJax, ParticlesJax

# use Tiger POMDP
from pomdp_problems.tiger.tiger_problem import (
    TigerState,
    TigerAction,
    TigerObservation,
    ObservationModel,
    TransitionModel,
    RewardModel,
    PolicyModel,
    TigerProblem,
    test_planner,
)


def init_problem():
    init_true_state = random.choice([
        TigerState("tiger-left"),
        TigerState("tiger-right"),
    ])
    # histogram representation
    init_belief = pomdp_py.Histogram({
        TigerState("tiger-left"): 0.5,
        TigerState("tiger-right"): 0.5,
    })
    # particle representation
    init_belief = pomdp_py.Particles([
        TigerState("tiger-left"),
        TigerState("tiger-right"),
    ])
    belief2 = ParticlesJax([
        TigerState("tiger-left"),
        TigerState("tiger-right"),
    ], np.array([1, 1]))
    tiger_problem = TigerProblem(
        0.15,  # observation noise
        init_true_state,
        init_belief,
    )
    planner = pomdp_py.POMCP(
        max_depth=3, discount_factor=1,
        planning_time=.5, exploration_const=110,
        rollout_policy=tiger_problem.agent.policy_model)
    tiger_problem2 = TigerProblem(
        0.15,  # observation noise
        init_true_state,
        init_belief,
    )
    planner2 = PomcpJax(
        max_depth=3, discount_factor=1,
        planning_time=.5, exploration_const=110,
        rollout_policy=tiger_problem2.agent.policy_model)
    return tiger_problem, tiger_problem2, init_true_state, init_belief, belief2, planner, planner2


def test_rollout():
    # Rollout gives a Monte Carlo estimate of the value function

    problem, problem2, state, belief, belief2, planner, planner2 = init_problem()
    # initialize things
    planner.plan(problem.agent)
    history = () # empty history
    root = None
    depth = 0
    ntrials = 1000
    total_reward = 0.
    for n in range(ntrials):
        total_reward += planner._rollout(state, history, root, depth)

    # batch over ntrials
    planner2.plan(problem2.agent)
    total_reward = planner2._rollout(state, history, root, depth, ntrials)

    # assert close enough

def test_plan():
    problem, problem2, state, belief, belief2, planner, planner2 = init_problem()
    planner = pomdp_py.POMCP(
        max_depth=3, discount_factor=1,
        planning_time=10, exploration_const=110,
        rollout_policy=problem.agent.policy_model)
    planner2 = PomcpJax(
        max_depth=3, discount_factor=1,
        planning_time=10, exploration_const=110,
        rollout_policy=problem2.agent.policy_model)

    planner.plan(problem.agent)
    planner2.plan(problem2.agent)

    # iterate over tree, check beliefs and values are close
    import pdb; pdb.set_trace()


def test_search():
    pass


test_rollout()
test_plan()
