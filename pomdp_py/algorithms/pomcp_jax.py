
import copy
import time
import random
import math

import jax.numpy as np

from pomdp_py.framework.basics import Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel
from pomdp_py.framework.planner import Planner
from pomdp_py.representations.distribution.particles import Particles
from pomdp_py.algorithms.po_uct import VNode, RootVNode, QNode, POUCT, RandomRollout
from pomdp_py.algorithms.pomcp import VNodeParticles, RootVNodeParticles, POMCP


def particle_reinvigoration(particles, num_particles, state_transform_func=None):
    """Note that particles should contain states that have already made
    the transition as a result of the real action. Therefore, they simply
    form part of the reinvigorated particles. At least maintain `num_particles`
    number of particles. If already have more, then it's ok.
    """
    # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
    new_particles = copy.deepcopy(particles)
    if len(new_particles) == 0:
        raise ValueError("Particle deprivation.")

    if len(new_particles) > num_particles:
        return new_particles
    
    print("Particle reinvigoration for %d particles" % (num_particles - len(new_particles)))
    while len(new_particles) < num_particles:
        # need to make a copy otherwise the transform affects states in 'particles'
        next_state = copy.deepcopy(particles.random())
        # Add artificial noise
        if state_transform_func is not None:
            next_state = state_transform_func(next_state)
        new_particles.add(next_state)
    return new_particles

def update_particles_belief(
    current_particles,
    real_action,
    real_observation=None,
    observation_model=None,
    transition_model=None,
    blackbox_model=None,
    state_transform_func=None,
):
    """
    update_particles_belief(Particles current_particles,
                           Action real_action, Observation real_observation=None,
                           ObservationModel observation_model=None,
                           TransitionModel transition_model=None,
                           BlackboxModel blackbox_model=None,
                           state_transform_func=None)
    This is the second case (update particles belief explicitly); Either
    BlackboxModel is not None, or TransitionModel and ObservationModel are not
    None. Note that you DON'T need to call this function if you are using POMCP.
    |TODO: not tested|
    Args:
        state_transform_func (State->State) is used to add artificial noise to
            the reinvigorated particles.
    """
    for particle in current_particles.particles:
        # particle represents a state
        if blackbox_model is not None:
            # We're using a blackbox generator; (s',o,r) ~ G(s,a)
            result = blackbox_model.sample(particle, real_action)
            next_state = result[0]
            observation = result[1]
        else:
            # We're using explicit models
            next_state = transition_model.sample(particle, real_action)
            observation = observation_model.sample(next_state, real_action)
        # If observation matches real, then the next_state is accepted
        if observation == real_observation:
            filtered_particles.append(next_state)
    # Particle reinvigoration
    return particle_reinvigoration(Particles(filtered_particles), len(current_particles.particles),
                                   state_transform_func=state_transform_func)

def sample_explicit_models(T, O, R, state, action, discount_factor=1.):
    # states, actions: batch, returns next_state, reward: batch
    next_state = T.sample(state, action)
    reward = R.sample(state, action, next_state)
    nsteps = 1
    if O is not None:
        observation = O.sample(next_state, action)
        return next_state, observation, reward, nsteps
    else:
        return next_state, reward, nsteps

class ParticlesJax(Particles):
    # represents a belief / distribution over states
    def __init__(self, values: List[State], weights: np.ndarray):
        self._values = values # used to convert from integer to State
        self._weights = weights # can be unnormalized, i.e. counts

    def add(self, particle, weight=1):
        # not sure we want to use this API
        self._weights = self._weights.at[particle].add(weight)
            #self._values.index(particle)
            #if isinstance(particle, State)
            #else particle
        #].add(weight)


class PomcpJax(POMCP):

    """POMCP is POUCT + particle belief representation.
    This POMCP version only works for problems
    with action space that can be enumerated."""

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(), action_prior=None,
                 show_progress=False, pbar_update_interval=5):
        super().__init__(max_depth=max_depth,
                         planning_time=planning_time,
                         num_sims=num_sims,
                         discount_factor=discount_factor,
                         exploration_const=exploration_const,
                         num_visits_init=num_visits_init,
                         value_init=value_init,
                         rollout_policy=rollout_policy,
                         action_prior=action_prior,
                         show_progress=show_progress,
                         pbar_update_interval=pbar_update_interval)


    def plan(self, agent):
        # Only works if the agent's belief is particles
        if not isinstance(agent.belief, ParticlesJax):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")
        return POUCT.plan(self, agent)

    def update(self, agent, real_action, real_observation,
                 state_transform_func=None):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.

        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, ParticlesJax):
            raise TypeError("agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")
        if not hasattr(agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation],
                                                   agent.history)
        tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(
            tree_belief,
            len(agent.init_belief.particles),
            state_transform_func=state_transform_func))
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)


    def _search(self):
        if self._show_progress:
            if stop_by_sims:
                total = int(self._num_sims)
            else:
                total = self._planning_time
            pbar = tqdm(total=total)

        start_time = time.time()
        while True:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._agent.sample_belief()
            self._simulate(state, self._agent.history, self._agent.tree,
                           None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time

            if self._show_progress and sims_count % self._pbar_update_interval == 0:
                if stop_by_sims:
                    pbar.n = sims_count
                else:
                    pbar.n = time_taken
                pbar.refresh()

            if stop_by_sims:
                if sims_count >= self._num_sims:
                    break
            else:
                if time_taken > self._planning_time:
                    if self._show_progress:
                        pbar.n = self._planning_time
                        pbar.refresh()
                    break

        if self._show_progress:
            pbar.close()

        best_action = self._agent.tree.argmax()
        return best_action, time_taken, sims_count

    def _simulate(self,
        state, history, root, parent,
        observation, depth):
        if depth > self._max_depth:
            return 0
        if root is None:
            if self._agent.tree is None:
                root = self._VNode(agent=self._agent, root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()
            if parent is not None:
                parent[observation] = root
            self._expand_vnode(root, history, state=state)
            rollout_reward = self._rollout(state, history, root, depth)
            return rollout_reward
        action = self._ucb(root)
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        if nsteps == 0:
            # This indicates the provided action didn't lead to transition
            # Perhaps the action is not allowed to be performed for the given state
            # (for example, the state is not in the initiation set of the option,
            # or the state is a terminal state)
            return reward

        total_reward = reward + (self._discount_factor**nsteps)*self._simulate(
            next_state,
            history + ((action, observation),),
            root[action][observation],
            root[action],
            observation,
            depth+nsteps)
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)

        # POMCP simulate, need to update belief as well
        if depth == 1 and root is not None:
            root.belief.add(state)  # belief update happens as simulation goes.
        return total_reward

    def _rollout(self, state, history, root, depth):
        while depth < self._max_depth:
            action = self._rollout_policy.rollout(state, history)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
        return total_discounted_reward

    def _ucb(self, root):
        """UCB1"""
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].value + \
                    self._exploration_const * math.sqrt(math.log(root.num_visits + 1) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    def _sample_generative_model(self, state, action):
        '''
        (s', o, r) ~ G(s, a)
        '''
        if self._agent.transition_model is None:
            next_state, observation, reward = self._agent.generative_model.sample(state, action)
        else:
            next_state = self._agent.transition_model.sample(state, action)
            observation = self._agent.observation_model.sample(next_state, action)
            reward = self._agent.reward_model.sample(state, action, next_state)
        return next_state, observation, reward

    def _VNode(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            # agent cannot be None.
            return RootVNodeParticles(self._num_visits_init,
                                      agent.history,
                                      belief=copy.deepcopy(agent.belief))
        else:
            if agent is None:
                return VNodeParticles(self._num_visits_init,
                                      belief=Particles([]))
            else:
                return VNodeParticles(self._num_visits_init,
                                      belief=copy.deepcopy(agent.belief))
