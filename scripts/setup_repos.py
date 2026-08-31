#!/usr/bin/env python3
"""Clone the external repositories this comparison needs, onto the right host.

Author: Dawit Chun

Two hosts, and the split matters. This machine is the training and harness host: it has the GPU,
no native ROS, and it already talks to the robot over Zenoh the way infer.py does. The robot PC
(the Orin) runs the VR stack and the motion controller inside Docker. Cloning the VR packages here
is for READING the source, not for running them.

Dry run is the default. Pass --run to actually clone.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXTERNAL = HERE / "external"

#Only what the two-way comparison actually needs. OpenPI and pi-0.5 are deliberately absent:
#HG-DAgger and our method share one base policy (act_prior), so no second model is required and
#the entire OpenPI install, its inference node and its checkpoint drop out of the plan.
REPOS = [
    dict(
        name="RLinf",
        url="https://github.com/RLinf/RLinf.git",
        branch=None,
        why="reference implementation of HG-DAgger: only_save_expert, only_success, the online "
            "LeRobot writer and the intervene_flag chunk sampler. Read for fidelity of the "
            "baseline; we are not porting its environment layer (see docs/AIWORKER_DAGGER_PORT.md).",
        host="this machine (reference)",
    ),
    dict(
        name="robotis_applications",
        url="https://github.com/ROBOTIS-GIT/robotis_applications.git",
        branch="jazzy",
        why="robotis_vuer, the Meta Quest VR publisher. Cloned here to read vr_publisher_sg2.py "
            "while writing the QuestIntervention shim; it must ALSO be cloned on the robot PC, "
            "where it is actually launched.",
        host="this machine (reference) AND the robot PC (to run)",
    ),
]

#Needed on the robot PC only. Listed so the requirement is written down, not so it is cloned here.
ROBOT_PC_ONLY = [
    ("ai_worker", "ffw_bringup, the robot bringup and hardware interface"),
    ("cyclo_control", "the motion-control layer; launched with controller_type:=vr"),
]


def run(cmd, dry):
    print("   $ " + " ".join(cmd))
    if dry:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="actually clone; default is a dry run")
    ap.add_argument("--depth", type=int, default=1, help="shallow clone depth, 0 for full history")
    a = ap.parse_args()
    dry = not a.run

    print(f"target: {EXTERNAL}")
    print(f"mode:   {'DRY RUN -- nothing will be written' if dry else 'CLONING'}\n")

    if not dry:
        EXTERNAL.mkdir(parents=True, exist_ok=True)

    ok = True
    for r in REPOS:
        dest = EXTERNAL / r["name"]
        print(f"[{r['name']}]  host: {r['host']}")
        print(f"   why: {r['why']}")
        if dest.exists():
            print(f"   already present at {dest} -- skipping\n")
            continue
        cmd = ["git", "clone"]
        if a.depth:
            cmd += ["--depth", str(a.depth)]
        if r["branch"]:
            cmd += ["-b", r["branch"]]
        cmd += [r["url"], str(dest)]
        if not run(cmd, dry):
            print(f"   FAILED to clone {r['name']}")
            ok = False
        print()

    print("On the ROBOT PC, not here:")
    for name, why in ROBOT_PC_ONLY:
        print(f"   {name:<22} {why}")
    print("   robotis_applications   clone there too, on branch jazzy, to launch robotis_vuer")

    if not dry:
        print("\nresult:")
        for r in REPOS:
            d = EXTERNAL / r["name"]
            n = sum(1 for _ in d.rglob("*")) if d.exists() else 0
            size = shutil.disk_usage(EXTERNAL).free // (1 << 30) if EXTERNAL.exists() else 0
            print(f"   {r['name']:<22} {'present' if d.exists() else 'MISSING'}  ({n} entries)")
        print(f"   disk free: {size} GB")
    else:
        print("\nnothing was written. rerun with --run to clone.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
