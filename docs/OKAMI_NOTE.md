# Single-video initialisation, and where our method attaches to it

Author: Dawit Chun
Parked on 2026-08-14. Not on the critical path; revisit only if the teleop and RL work finishes early.

## The paper

OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation.
Li, Zhu, Xie, Jiang, Seo, Pavlakos, Zhu. arXiv:2410.11792.

One RGB-D video of a human doing the task. Open-world vision models identify the task-relevant
objects. Body motion and hand poses are retargeted **separately**, with object-aware retargeting
adapting the motion to wherever the objects actually are. Reported average success 79.2%.

## Correct the premise before building on it

OKAMI does **not** stop at a splined trajectory. The retargeted plan is used to generate rollouts,
and those rollouts train a closed-loop visuomotor policy. "Replace the splining with something
learned" is, as stated, already the paper. Any proposal has to start from that.

## Where the opening actually is

The plan generation is geometric, and the learned policy is trained by behaviour cloning on what
that geometry produced. The pipeline is

    video -> object-aware retargeting -> rollouts -> BC

and there is no mechanism anywhere in it to **exceed** the retargeted demonstration. If the
retargeting is 80% right, a policy cloned from it caps out near 80%. The reported 79.2% is
consistent with that being the binding constraint rather than the policy class.

## Why this is complementary to our work rather than competing with it

The two halves answer different questions and do not overlap:

* OKAMI answers *where do demonstrations come from* -- one video, no teleoperation.
* Ours answers *how do you improve past the demonstrations cheaply* -- the servo reward supplies an
  absolute task signal, the human gate supplies dense relative shaping as a **potential** (see
  HIL_THEORY.md, Theorem 3), and the human stops being needed once the gate classifier is fitted.

Stacked: **one human video to initialise, then human-gated RL to exceed it, with no learned reward
model anywhere and the human queried under 10% of the time.**

The alternative framing -- swap the geometric retargeting for a learned or VLA module -- is a
perception project, it is crowded, and it leverages none of the theory or hardware work already
done. Not recommended.

## Hardware caveat, and it is a real one

OKAMI retargets body motion and hand poses separately, and the hand-pose branch presupposes a hand
with more than one degree of freedom. The AI Worker gripper has one. Roughly half the method has
nothing to retarget onto this robot, so a port would carry the body-motion half only. Not fatal,
but less of the paper transfers than it first appears.

## Task fit

The held-container task (one arm holds a bottle or bin, the other deposits objects into it) suits
both halves. It is the kind of task a single human video can bootstrap, and the last few
millimetres -- the alignment over the mouth, which is exactly what retargeting gets wrong -- is
what the RL loop then fixes.

It also has a property worth recording separately, because it applies well beyond OKAMI:

> **When one arm holds the target, the target's pose is known exactly from forward kinematics.**

The robot knows where the container mouth is because it is holding the container. So "was the right
gripper over the opening when it released?" is computable from joint angles alone -- no vision, no
learned success classifier. The same argument gives the base pose for free in a bimanual assembly
task. The error sources are kinematic calibration and **slip in the gripper**, not perception.

## What to measure before any of this is worth pursuing

1. FK-relative accuracy between the two arms, against a ~25 mm bottle mouth. Slip is the thing that
   would break it.
2. Whether a deposit registers on the holding arm at all (gravity-torque step vs the 1.7 rad/1000
   following-error floor measured in the demos).

Both are read-only measurements and settle the bottle task, the assembly task, and this port at
once.

---

## The external-camera critique, and a better answer than fixing it

OKAMI records the human with an external RGB-D rig. That means the demonstration lives in a
different frame from the robot's own observations, so the extrinsics have to be calibrated and the
retargeting has to bridge a viewpoint gap. It also means the method is not something you can carry
into a new room.

### Reframe: the robot watches with its own cameras, and tracks OBJECTS, not the human

OKAMI retargets body motion and hand poses. But what a task *is* -- what has to be reproduced -- is
what happened to the objects. A human hand and a one-DoF parallel gripper produce the SAME object
trajectory through completely different arm motions. Object-centric imitation is therefore
embodiment-agnostic by construction, which is exactly the problem the separate body/hand retargeting
branches exist to solve.

On this robot that is decisive rather than merely tidy: the hand-pose branch has nothing to map onto
a parallel gripper, so half of OKAMI is dead on arrival here, while the object trajectory transfers
unchanged.

Two properties make the robot's own cameras better rather than just cheaper:

* **The camera pose is known from FK.** An external rig has to be calibrated against the robot; the
  robot's cameras are already in its kinematic chain. The calibration problem is removed, not solved.
* **Demonstration and execution share an observation space** -- same cameras, same lighting, same
  viewpoint distribution. There is no domain gap to bridge because it is literally the same sensor.

### The join with our theory: the video is a POTENTIAL, not a plan

OKAMI's ceiling exists because it clones a geometric plan; the policy cannot exceed the retargeting.
If instead the observed object trajectory defines Phi, then by Theorem 3 (HIL_THEORY.md sec 10) it
shapes learning on top of r_phys while leaving the optimum **exactly** unchanged. A blurry video, an
occluded object, an inefficient human, a different embodiment -- every one of those costs sample
efficiency and none of them costs correctness. It is the same guarantee that already covers the
human gate, applied to a second information source.

Full story: **the robot watches a human once through its own eyes, extracts what happened to the
objects, uses that as a potential, and the servo reward keeps it honest.** No external rig, no
retargeting, no behaviour-cloning ceiling, no learned reward model.

### Hard parts, honestly

1. Occlusion is worst exactly where it matters -- the object is inside the hand during the grasp.
   Tracking hand-and-object as one entity through that window is the likely answer and is real work.
2. Metric depth. Object trajectories need scale. Whether the scene camera provides depth, or stereo
   across the wrist cameras is required, decides the difficulty. Not yet checked.
3. Depends on a 3D object detector running on the robot's cameras -- already the plan for bottle
   pose, so shared infrastructure rather than a new dependency.
