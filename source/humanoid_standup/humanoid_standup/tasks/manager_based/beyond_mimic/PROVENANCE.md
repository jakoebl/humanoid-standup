This task adapts the Isaac Lab implementation from
https://github.com/HybridRobotics/whole_body_tracking at
`cd65172032893724b445448818c34165846d847d` (declared MIT in upstream setup.py).
The motion command, tracking reward/termination formulas, privileged observations,
and configuration structure originate there. Customizations match the user's
local mujocolab/mjlab G1 task: a reference-free 93-value actor, 286-value critic,
motor gains/action scales, 2048 training environments, and no joint-limit reward.

PhysX and MuJoCo are different dynamics engines. This task uses Isaac Lab's G1 USD
with its finger links/default finger actuators. Self-contact is measured on the
30 main G1 bodies with robot-only contact filters; ground contacts are excluded.
Finger-only self-contacts are not counted. Foot friction uses PhysX material
buckets rather than MuJoCo geom friction. Reference targets must be regenerated
on this asset, with name metadata, using scripts/precompute_beyond_mimic.py.
