from pomdp_py.algorithms.po_uct cimport VNode, RootVNode, POUCT
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.framework.basics cimport Agent, Action, Observation

cdef class VNodeParticles(VNode):
    cdef public Particles belief
cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

cdef class POMCP(POUCT):
    cpdef public force_expansion(POMCP self, Action action, Observation observation, int depth=*)
