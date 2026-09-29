#!/usr/bin/env python3
"""Estimate how long a job will sit in the Leonardo queue.

Three independent estimators are reported side by side rather than blended, because they
fail in different ways and their disagreement is itself informative:

  (a) backfill oracle    -- SLURM's own reservation (``squeue --start`` / ``sbatch --test-only``).
                            Exact when present, but only the top few percent of the queue has one.
  (b) empirical survival -- Kaplan-Meier over comparable historical jobs from ``sacct -a``.
  (c) queue-drain sim    -- rank simulated priority among genuinely-contending pending jobs,
                            then drain node-hours ahead at the partition's observed rate.

Stdlib only; everything comes from shelling out to the SLURM CLIs.

Usage
-----
    queue_eta.py --nodes 4 --time 12:00:00
    queue_eta.py --job 58373851
    queue_eta.py --script LeonardoRunScripts/grid_run_3.sh
    queue_eta.py --mine
    queue_eta.py --self-test
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DEFAULT_PARTITION = "boost_usr_prod"
DEFAULT_HISTORY_DAYS = 14
CACHE_TTL_SECONDS = 6 * 3600

# One high-throughput user can swamp the sample: over 30 days a single account submitted
# 1.67M of the 2.08M single-node `normal` jobs on boost_usr_prod, pulling the median wait
# down to ~2s and hiding what a normal submitter actually experiences. Cap each user's
# contribution so the distribution reflects users, not submission volume.
PER_USER_CAP = 300

# Reasons that mean a pending job is actually competing for the resources we want. Everything
# else (Dependency, JobHeldUser, BeginTime, InvalidAccount, ...) is parked for reasons unrelated
# to priority and must not be counted as being "ahead" of us -- on boost_usr_prod that is ~45%
# of the pending list.
CONTENDING_REASONS = frozenset({"Priority", "Resources", "None", ""})

# Node-count buckets used to pool comparable historical jobs. Chosen to match the scheduling
# regimes on this cluster rather than round numbers: 1 backfills freely, 2-16 is awkward,
# 17-64 is the dead zone below the bprod floor, 65+ is bprod territory.
NODE_BUCKETS: list[tuple[int, int, str]] = [
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 16, "5-16"),
    (17, 64, "17-64"),
    (65, 10**9, ">64"),
]

# Walltime buckets in hours.
TIME_BUCKETS: list[tuple[float, float, str]] = [
    (0.0, 0.5, "<=30m"),
    (0.5, 2.0, "30m-2h"),
    (2.0, 8.0, "2-8h"),
    (8.0, 16.0, "8-16h"),
    (16.0, 1e9, ">16h"),
]

MIN_SAMPLES = 30


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def run(cmd: Sequence[str], timeout: int = 120) -> str:
    """Run a command, returning stdout. Returns '' on failure rather than raising."""
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout


def parse_walltime(text: str | None) -> float | None:
    """Parse a SLURM duration ([DD-]HH:MM:SS, MM:SS, or plain minutes) into seconds."""
    if not text:
        return None
    text = text.strip()
    if text in {"UNLIMITED", "Partition_Limit", "NOT_SET", "None", ""}:
        return None
    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    elif len(nums) == 1:
        # Bare number is minutes, per sbatch --time semantics.
        h, m, s = 0.0, nums[0], 0.0
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def fmt_duration(seconds: float | None) -> str:
    """Human-readable duration, tuned for reading queue waits at a glance."""
    if seconds is None:
        return "n/a"
    if seconds < 0:
        seconds = 0.0
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m"
    hours = seconds / 3600
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def parse_dt(text: str | None) -> dt.datetime | None:
    if not text or text in {"Unknown", "None", "N/A", ""}:
        return None
    try:
        return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def bucket_of(value: float, buckets: Iterable[tuple[Any, Any, str]]) -> str:
    for lo, hi, label in buckets:
        if lo <= value <= hi:
            return label
    return "?"


def parse_tres(text: str | None) -> dict[str, float]:
    """Parse a SLURM TRES string like 'cpu=32,mem=100G,node=4' into a dict."""
    out: dict[str, float] = {}
    if not text or text in {"", "(null)"}:
        return out
    for item in text.split(","):
        if "=" not in item:
            continue
        key, _, raw = item.partition("=")
        out[key.strip()] = parse_memory(raw.strip())
    return out


_MEM_UNITS = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}


def parse_memory(raw: str) -> float:
    """Parse a possibly unit-suffixed number ('100G', '4400', '1.5T')."""
    if not raw:
        return 0.0
    match = re.fullmatch(r"([0-9.]+)\s*([KMGTP]?)", raw.strip(), re.IGNORECASE)
    if not match:
        try:
            return float(raw)
        except ValueError:
            return 0.0
    value = float(match.group(1))
    suffix = match.group(2).upper()
    return value * _MEM_UNITS.get(suffix, 1.0)


# --------------------------------------------------------------------------------------
# Cluster configuration probes
# --------------------------------------------------------------------------------------


@dataclass
class PriorityConfig:
    weight_age: float = 0.0
    weight_fairshare: float = 0.0
    weight_qos: float = 0.0
    weight_jobsize: float = 0.0
    weight_partition: float = 0.0
    max_age_seconds: float = 7 * 86400
    flags: frozenset[str] = frozenset()
    scheduler_params: dict[str, str] = field(default_factory=dict)
    usage_reset_period: str = ""
    # SLURM normalises job size against the whole cluster, not the partition. On Leonardo
    # that is MaxNodeCount=5031, not boost_usr_prod's 3264 -- using the partition total
    # inflates the job-size term by ~1.54x.
    cluster_nodes: int = 0

    @property
    def small_relative_to_time(self) -> bool:
        return "SMALL_RELATIVE_TO_TIME" in self.flags

    @property
    def max_tres(self) -> bool:
        return "MAX_TRES" in self.flags

    @property
    def favor_small(self) -> bool:
        return "FAVOR_SMALL" in self.flags

    @property
    def bf_max_job_user_part(self) -> int | None:
        raw = self.scheduler_params.get("bf_max_job_user_part")
        return int(raw) if raw and raw.isdigit() else None


def read_priority_config() -> PriorityConfig:
    text = run(["scontrol", "show", "config"])
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()

    def num(key: str) -> float:
        raw = values.get(key, "0")
        try:
            return float(raw)
        except ValueError:
            return 0.0

    sched: dict[str, str] = {}
    for item in values.get("SchedulerParameters", "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.partition("=")
        sched[key.strip()] = val.strip()

    flags = frozenset(
        f.strip() for f in values.get("PriorityFlags", "").split(",") if f.strip()
    )
    max_age = parse_walltime(values.get("PriorityMaxAge")) or 7 * 86400
    if values.get("PriorityFavorSmall", "no").lower() == "yes":
        flags = flags | {"FAVOR_SMALL"}

    return PriorityConfig(
        weight_age=num("PriorityWeightAge"),
        weight_fairshare=num("PriorityWeightFairShare"),
        weight_qos=num("PriorityWeightQOS"),
        weight_jobsize=num("PriorityWeightJobSize"),
        weight_partition=num("PriorityWeightPartition"),
        max_age_seconds=max_age,
        flags=flags,
        scheduler_params=sched,
        usage_reset_period=values.get("PriorityUsageResetPeriod", ""),
        cluster_nodes=int(num("MaxNodeCount")),
    )


@dataclass
class QosInfo:
    name: str
    priority: float = 0.0
    max_wall: float | None = None
    max_tres: dict[str, float] = field(default_factory=dict)
    max_tres_pu: dict[str, float] = field(default_factory=dict)
    grp_tres: dict[str, float] = field(default_factory=dict)
    max_jobs_pu: int | None = None
    max_submit_pu: int | None = None
    flags: frozenset[str] = frozenset()


def read_qos() -> dict[str, QosInfo]:
    text = run(
        [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "format=Name,Priority,MaxWall,MaxTRES,MaxTRESPU,GrpTRES,MaxJobsPU,MaxSubmitPU,Flags",
        ]
    )
    out: dict[str, QosInfo] = {}
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 9 or not cols[0]:
            continue

        def as_int(raw: str) -> int | None:
            return int(raw) if raw.strip().isdigit() else None

        try:
            priority = float(cols[1]) if cols[1] else 0.0
        except ValueError:
            priority = 0.0
        out[cols[0]] = QosInfo(
            name=cols[0],
            priority=priority,
            max_wall=parse_walltime(cols[2]),
            max_tres=parse_tres(cols[3]),
            max_tres_pu=parse_tres(cols[4]),
            grp_tres=parse_tres(cols[5]),
            max_jobs_pu=as_int(cols[6]),
            max_submit_pu=as_int(cols[7]),
            flags=frozenset(f.strip() for f in cols[8].split(",") if f.strip()),
        )
    return out


@dataclass
class PartitionInfo:
    name: str
    total_nodes: int = 0
    total_cpus: int = 0
    total_gpus: int = 0
    max_nodes: int | None = None
    max_time: float | None = None
    default_time: float | None = None
    default_qos: str = ""


def read_partitions() -> dict[str, PartitionInfo]:
    text = run(["scontrol", "show", "partition", "-o"])
    out: dict[str, PartitionInfo] = {}
    for line in text.splitlines():
        fields = dict(
            (m.group(1), m.group(2))
            for m in re.finditer(r"(\w+)=(\S*)", line)
            if m.group(1)
        )
        name = fields.get("PartitionName")
        if not name:
            continue
        tres = parse_tres(fields.get("TRES"))

        def as_int(key: str) -> int | None:
            raw = fields.get(key, "")
            return int(raw) if raw.isdigit() else None

        out[name] = PartitionInfo(
            name=name,
            total_nodes=int(tres.get("node", 0)),
            total_cpus=int(tres.get("cpu", 0)),
            total_gpus=int(tres.get("gres/gpu", 0)),
            max_nodes=as_int("MaxNodes"),
            max_time=parse_walltime(fields.get("MaxTime")),
            default_time=parse_walltime(fields.get("DefaultTime")),
            default_qos=fields.get("QoS", ""),
        )
    return out


def read_associations(user: str) -> list[tuple[str, frozenset[str]]]:
    """Return (account, allowed-qos) pairs for the user."""
    # 'where' and 'user=<x>' must be separate argv elements; passed as one string sacctmgr
    # silently returns nothing, which would leave us with no QoS options at all.
    text = run(
        ["sacctmgr", "-nP", "show", "assoc", "where", f"user={user}", "format=Account,QOS"]
    )
    out: list[tuple[str, frozenset[str]]] = []
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 2 or not cols[0]:
            continue
        out.append((cols[0], frozenset(q for q in cols[1].split(",") if q)))
    return out


def read_fairshare(user: str) -> dict[str, float]:
    """Map account -> fairshare factor in [0, 1]."""
    text = run(["sshare", "-nP", "-U", "-o", "Account,User,FairShare"])
    out: dict[str, float] = {}
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 3 or not cols[0]:
            continue
        try:
            # Accounts are case-insensitive in SLURM and scripts here write them uppercase
            # (#SBATCH --account=ICT26_MHPC_0) while sshare reports them lowercase.
            out[cols[0].lower()] = float(cols[2])
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------------------
# Priority simulation
# --------------------------------------------------------------------------------------


@dataclass
class JobSpec:
    nodes: int
    walltime: float  # seconds
    qos: str
    account: str
    partition: str
    jobid: str | None = None
    name: str = ""
    gpus_per_node: int = 4  # Leonardo booster nodes carry 4 GPUs.

    @property
    def node_hours(self) -> float:
        return self.nodes * self.walltime / 3600.0


@dataclass
class PriorityBreakdown:
    total: float
    age: float
    fairshare: float
    qos: float
    jobsize: float
    partition: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "age": self.age,
            "fairshare": self.fairshare,
            "qos": self.qos,
            "jobsize": self.jobsize,
            "partition": self.partition,
        }


class PriorityModel:
    """Reproduces SLURM's multifactor priority for this cluster.

    The weights are misleading read naively: PriorityWeightJobSize is 1e7, a hundred times
    the QOS weight, yet the observed JOBSIZE term is 1-400. SMALL_RELATIVE_TO_TIME is why --
    it divides the size fraction by the walltime in minutes, so the term only becomes
    meaningful for a huge job with a very short limit. QOS dominates in practice.
    """

    def __init__(
        self,
        config: PriorityConfig,
        qos_table: dict[str, QosInfo],
        partitions: dict[str, PartitionInfo],
        fairshare: dict[str, float],
    ) -> None:
        self.config = config
        self.qos_table = qos_table
        self.partitions = partitions
        self.fairshare = fairshare
        # SLURM normalises QOS priority against the highest QOS priority defined anywhere.
        self.max_qos_priority = max(
            (q.priority for q in qos_table.values()), default=0.0
        ) or 1.0

    def jobsize_factor(self, spec: JobSpec) -> float:
        # Verified against sprio: the denominator is the cluster-wide node count, so a
        # 256-node/24h job gives 1e7*(256/5031)/1440 = 353.4, matching sprio's 353 exactly.
        total = self.config.cluster_nodes
        if not total:
            part = self.partitions.get(spec.partition)
            total = part.total_nodes if part else 0
        if not total:
            return 0.0

        factor = spec.nodes / total

        if self.config.small_relative_to_time:
            minutes = max(spec.walltime / 60.0, 1.0)
            factor /= minutes
        elif self.config.favor_small:
            factor = 1.0 - factor

        return min(max(factor, 0.0), 1.0)

    def compute(self, spec: JobSpec, age_seconds: float = 0.0) -> PriorityBreakdown:
        cfg = self.config
        qos = self.qos_table.get(spec.qos)
        qos_norm = (qos.priority / self.max_qos_priority) if qos else 0.0

        age_norm = min(age_seconds / cfg.max_age_seconds, 1.0) if cfg.max_age_seconds else 0.0
        fs = self.fairshare.get(spec.account.lower(), 0.0)

        age = cfg.weight_age * age_norm
        fairshare = cfg.weight_fairshare * fs
        qos_term = cfg.weight_qos * qos_norm
        jobsize = cfg.weight_jobsize * self.jobsize_factor(spec)
        partition = 0.0  # PriorityWeightPartition is 0 on Leonardo.

        total = age + fairshare + qos_term + jobsize + partition
        return PriorityBreakdown(
            total=math.floor(total),
            age=age,
            fairshare=fairshare,
            qos=qos_term,
            jobsize=jobsize,
            partition=partition,
        )


# --------------------------------------------------------------------------------------
# Live queue
# --------------------------------------------------------------------------------------


@dataclass
class PendingJob:
    jobid: str
    user: str
    nodes: int
    walltime: float
    priority: float
    reason: str
    qos: str

    @property
    def contending(self) -> bool:
        return self.reason in CONTENDING_REASONS

    @property
    def node_hours(self) -> float:
        return self.nodes * self.walltime / 3600.0


@dataclass
class RunningJob:
    jobid: str
    nodes: int
    remaining: float  # seconds until the time limit expires


@dataclass
class QueueState:
    pending: list[PendingJob]
    running: list[RunningJob]
    partition: PartitionInfo

    @property
    def contending(self) -> list[PendingJob]:
        return [j for j in self.pending if j.contending]

    def drain_rate(self) -> float:
        """Node-hours of queued work the partition can absorb per wall-clock hour.

        A partition of N nodes delivers N node-hours per hour when fully packed, so that is
        the ceiling. Real packing is imperfect (jobs do not tile exactly, and time limits
        over-state actual runtimes in the opposite direction), so scale by the observed
        utilisation as a rough efficiency factor.
        """
        total = float(self.partition.total_nodes or 0)
        if total <= 0:
            return 1.0
        busy = sum(j.nodes for j in self.running)
        utilisation = min(busy / total, 1.0) if busy else 0.9
        return total * max(utilisation, 0.5)


def read_queue(partition: str) -> QueueState:
    partitions = read_partitions()
    part = partitions.get(partition, PartitionInfo(name=partition))

    pending: list[PendingJob] = []
    text = run(
        [
            "squeue",
            "-p",
            partition,
            "-t",
            "PD",
            "-h",
            "-o",
            "%i|%u|%D|%l|%Q|%r|%q",
        ]
    )
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 7:
            continue
        try:
            nodes = int(cols[2])
            priority = float(cols[4])
        except ValueError:
            continue
        pending.append(
            PendingJob(
                jobid=cols[0],
                user=cols[1],
                nodes=nodes,
                walltime=parse_walltime(cols[3]) or 0.0,
                priority=priority,
                reason=cols[5].strip(),
                qos=cols[6],
            )
        )

    running: list[RunningJob] = []
    text = run(["squeue", "-p", partition, "-t", "R", "-h", "-o", "%i|%D|%L"])
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 3:
            continue
        try:
            nodes = int(cols[1])
        except ValueError:
            continue
        running.append(
            RunningJob(
                jobid=cols[0], nodes=nodes, remaining=parse_walltime(cols[2]) or 0.0
            )
        )

    return QueueState(pending=pending, running=running, partition=part)


# --------------------------------------------------------------------------------------
# Historical data + Kaplan-Meier
# --------------------------------------------------------------------------------------

HISTORY_FIELDS = [
    "JobIDRaw",
    "User",
    "Account",
    "QOS",
    "Partition",
    "NNodes",
    "Timelimit",
    "Submit",
    "Eligible",
    "Start",
    "End",
    "State",
    "Priority",
]


@dataclass
class HistoricalJob:
    qos: str
    nodes: int
    walltime: float
    wait: float  # seconds from eligible to start (or to censoring time)
    censored: bool
    user: str = ""
    priority: float = 0.0


def cache_dir() -> Path:
    env = os.environ.get("QUEUE_ETA_CACHE")
    if env:
        return Path(env)
    for base in ("/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs", os.environ.get("FAST", "")):
        if base and Path(base).is_dir():
            return Path(base) / "queue_eta_cache"
    return Path.home() / ".cache" / "queue_eta"


def fetch_history(partition: str, days: int, refresh: bool) -> list[HistoricalJob]:
    """Load historical jobs from cache, refreshing from sacct when stale."""
    cdir = cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"hist_{partition}_{days}d.csv.gz"

    stale = refresh or not path.exists()
    if not stale:
        age = time.time() - path.stat().st_mtime
        stale = age > CACHE_TTL_SECONDS

    if stale:
        text = run(
            [
                "sacct",
                "-a",
                "-X",
                "-P",
                "-n",
                "-r",
                partition,
                "-S",
                f"now-{days}days",
                "-E",
                "now",
                "-o",
                ",".join(HISTORY_FIELDS),
            ],
            timeout=600,
        )
        if text:
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as fh:
                fh.write(text)
            tmp.replace(path)
        elif not path.exists():
            return []

    # Parsing ~1M timestamps costs ~25s, so keep a derived cache of just what the estimator
    # needs and reuse it while the raw pull stands. The v2 suffix invalidates caches written
    # before the priority column existed.
    derived = cdir / f"derived_v2_{partition}_{days}d.csv.gz"
    if derived.exists() and derived.stat().st_mtime >= path.stat().st_mtime:
        out = []
        with gzip.open(derived, "rt") as fh:
            for cols in csv.reader(fh):
                if len(cols) < 7:
                    continue
                out.append(
                    HistoricalJob(
                        qos=cols[0],
                        nodes=int(cols[1]),
                        walltime=float(cols[2]),
                        wait=float(cols[3]),
                        censored=cols[4] == "1",
                        user=cols[5],
                        priority=float(cols[6]),
                    )
                )
        return out

    now = dt.datetime.now()
    out: list[HistoricalJob] = []
    with gzip.open(path, "rt") as fh:
        for cols in csv.reader(fh, delimiter="|"):
            if len(cols) < len(HISTORY_FIELDS):
                continue
            row = dict(zip(HISTORY_FIELDS, cols))
            try:
                nodes = int(row["NNodes"])
            except ValueError:
                continue
            walltime = parse_walltime(row["Timelimit"])
            if not walltime:
                continue

            eligible = parse_dt(row["Eligible"]) or parse_dt(row["Submit"])
            if eligible is None:
                continue
            start = parse_dt(row["Start"])
            state = row["State"].split()[0] if row["State"] else ""

            if start is not None:
                wait = (start - eligible).total_seconds()
                censored = False
            elif state in {"PENDING", "REQUEUED", "SUSPENDED"}:
                # Still waiting: a right-censored observation. Dropping these would bias
                # every estimate low, because the long waits are exactly the ones still queued.
                wait = (now - eligible).total_seconds()
                censored = True
            else:
                continue

            if wait < 0:
                continue
            out.append(
                HistoricalJob(
                    qos=row["QOS"],
                    nodes=nodes,
                    walltime=walltime,
                    wait=wait,
                    censored=censored,
                    user=row["User"],
                    priority=float(row["Priority"]) if row["Priority"].strip() else 0.0,
                )
            )

    tmp = derived.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as fh:
        writer = csv.writer(fh)
        for j in out:
            writer.writerow(
                [
                    j.qos,
                    j.nodes,
                    f"{j.walltime:.0f}",
                    f"{j.wait:.0f}",
                    int(j.censored),
                    j.user,
                    f"{j.priority:.0f}",
                ]
            )
    tmp.replace(derived)
    return out


def cap_per_user(samples: Sequence[HistoricalJob], cap: int = PER_USER_CAP) -> list[HistoricalJob]:
    """Limit each user's contribution so one high-volume submitter cannot define the median.

    Takes an evenly-spaced slice of each user's jobs rather than the first `cap`, so the
    sample still spans the whole window instead of whatever the user did on day one.
    """
    by_user: dict[str, list[HistoricalJob]] = defaultdict(list)
    for job in samples:
        by_user[job.user].append(job)

    out: list[HistoricalJob] = []
    for jobs in by_user.values():
        if len(jobs) <= cap:
            out.extend(jobs)
        else:
            stride = len(jobs) / cap
            out.extend(jobs[int(i * stride)] for i in range(cap))
    return out


def kaplan_meier(samples: Sequence[HistoricalJob]) -> list[tuple[float, float]]:
    """Return [(time, survival)] -- the probability of *still waiting* past each time."""
    events = sorted((s.wait, s.censored) for s in samples)
    n_at_risk = len(events)
    curve: list[tuple[float, float]] = []
    survival = 1.0
    i = 0
    while i < len(events):
        t = events[i][0]
        deaths = 0
        removed = 0
        while i < len(events) and events[i][0] == t:
            if not events[i][1]:
                deaths += 1
            removed += 1
            i += 1
        if n_at_risk > 0 and deaths > 0:
            survival *= 1.0 - deaths / n_at_risk
            curve.append((t, survival))
        n_at_risk -= removed
    return curve


def km_quantile(
    curve: Sequence[tuple[float, float]], q: float, elapsed: float = 0.0
) -> float | None:
    """Quantile of the wait, conditional on having already waited `elapsed` seconds.

    With elapsed=0 this is the plain quantile: the smallest t where survival drops to 1-q.

    With elapsed>0 it answers the question that actually matters for a job already sitting
    in the queue -- given it has survived this long, how much longer? That is
    P(W > t | W > elapsed) = S(t)/S(elapsed), and the answer is returned as residual time
    (t - elapsed). The distinction is not academic: single-node `normal` jobs here are
    bimodal, almost all backfilling within minutes while a minority stick for days, so the
    unconditional p50 of ~4 minutes is meaningless for a job that has already waited a day.

    Returns None when the curve never reaches the target, i.e. censoring means we cannot
    bound that quantile from the data.
    """
    s_elapsed = 1.0
    if elapsed > 0:
        for t, s in curve:
            if t <= elapsed:
                s_elapsed = s
            else:
                break
        if s_elapsed <= 0:
            return None

    target = s_elapsed * (1.0 - q)
    for t, s in curve:
        if t >= elapsed and s <= target:
            return t - elapsed
    return None


@dataclass
class EmpiricalEstimate:
    n: int
    n_censored: int
    p25: float | None
    p50: float | None
    p75: float | None
    p90: float | None
    bucket: str
    widened: bool
    n_users: int = 0
    n_raw: int = 0
    elapsed: float = 0.0
    priority_filtered: bool = False


def empirical_estimate(
    history: Sequence[HistoricalJob],
    spec: JobSpec,
    qos_table: dict[str, QosInfo],
    elapsed: float = 0.0,
    priority: float | None = None,
) -> EmpiricalEstimate:
    """Kaplan-Meier quantiles over comparable jobs, widening the pool if it is too thin.

    `elapsed` is how long this job has already waited; quantiles are then residual (how much
    longer), conditional on having got this far. `priority` narrows the pool to jobs that
    queued at a similar priority, which matters mostly in the tail -- within single-node
    `normal` jobs the bottom priority band has a p90 of ~13h against ~0.8h for the top.
    """
    node_bucket = bucket_of(spec.nodes, NODE_BUCKETS)
    time_bucket = bucket_of(spec.walltime / 3600.0, TIME_BUCKETS)

    def select(match_time: bool, match_qos_exact: bool) -> list[HistoricalJob]:
        target_prio = qos_table[spec.qos].priority if spec.qos in qos_table else None
        out = []
        for h in history:
            if bucket_of(h.nodes, NODE_BUCKETS) != node_bucket:
                continue
            if match_qos_exact:
                if h.qos != spec.qos:
                    continue
            elif target_prio is not None:
                # Fall back to pooling QoS of the same priority tier.
                hp = qos_table[h.qos].priority if h.qos in qos_table else None
                if hp is None or abs(hp - target_prio) > 1e-9:
                    continue
            if match_time and bucket_of(h.walltime / 3600.0, TIME_BUCKETS) != time_bucket:
                continue
            out.append(h)
        return out

    widened = False
    label = f"qos={spec.qos}, nodes={node_bucket}, time={time_bucket}"
    samples = select(match_time=True, match_qos_exact=True)
    if len(samples) < MIN_SAMPLES:
        widened = True
        samples = select(match_time=False, match_qos_exact=True)
        label = f"qos={spec.qos}, nodes={node_bucket}, any walltime"
    if len(samples) < MIN_SAMPLES:
        samples = select(match_time=False, match_qos_exact=False)
        label = f"qos-tier of {spec.qos}, nodes={node_bucket}, any walltime"

    if not samples:
        return EmpiricalEstimate(0, 0, None, None, None, None, label, widened)

    n_raw = len(samples)
    samples = cap_per_user(samples)

    # Narrow to jobs that queued at a comparable priority. Keep the band wide (a third of
    # the pool) so the sample stays usable, and skip it entirely if that would leave too
    # little to fit a curve.
    priority_filtered = False
    if priority is not None:
        ranked = sorted(samples, key=lambda s: s.priority)
        below = sum(1 for s in ranked if s.priority <= priority)
        pct = below / len(ranked)
        lo = max(0.0, min(pct - 1 / 6, 2 / 3))
        band = ranked[int(lo * len(ranked)) : int((lo + 1 / 3) * len(ranked))]
        if len(band) >= MIN_SAMPLES:
            samples = band
            priority_filtered = True
            label += f", priority ~{pct:.0%}ile"

    curve = kaplan_meier(samples)
    return EmpiricalEstimate(
        n=len(samples),
        n_censored=sum(1 for s in samples if s.censored),
        p25=km_quantile(curve, 0.25, elapsed),
        p50=km_quantile(curve, 0.50, elapsed),
        p75=km_quantile(curve, 0.75, elapsed),
        p90=km_quantile(curve, 0.90, elapsed),
        bucket=label,
        widened=widened,
        n_users=len({s.user for s in samples}),
        n_raw=n_raw,
        elapsed=elapsed,
        priority_filtered=priority_filtered,
    )


# --------------------------------------------------------------------------------------
# Estimator (c): queue-drain simulation
# --------------------------------------------------------------------------------------


@dataclass
class DrainEstimate:
    rank: int
    contending: int
    node_hours_ahead: float
    drain_rate: float
    eta: float | None


def drain_estimate(
    queue: QueueState, model: PriorityModel, spec: JobSpec, max_hours: float = 24 * 14
) -> DrainEstimate:
    """Walk simulated time forward, letting our job's age raise it past static-priority jobs.

    Our priority grows at weight_age/max_age per second while the jobs ahead of us are
    treated as static, so the rank we have to clear shrinks over time. We advance in steps,
    draining node-hours at the partition's observed rate until the backlog ahead is gone.
    """
    contenders = queue.contending
    if not contenders:
        return DrainEstimate(0, 0, 0.0, queue.drain_rate(), 0.0)

    ranked = sorted(contenders, key=lambda j: j.priority)
    priorities = [j.priority for j in ranked]
    # Cumulative node-hours from the *top* of the queue down to each position.
    suffix_nh: list[float] = [0.0] * (len(ranked) + 1)
    for i in range(len(ranked) - 1, -1, -1):
        suffix_nh[i] = suffix_nh[i + 1] + ranked[i].node_hours

    rate = queue.drain_rate()
    if rate <= 0:
        rate = 1.0

    age_rate = (
        model.config.weight_age / model.config.max_age_seconds
        if model.config.max_age_seconds
        else 0.0
    )

    initial = model.compute(spec, age_seconds=0.0).total
    idx = bisect.bisect_right(priorities, initial)
    rank0 = len(ranked) - idx
    nh0 = suffix_nh[idx]

    elapsed = 0.0
    step = 900.0  # 15 min
    drained = 0.0
    while elapsed < max_hours * 3600:
        prio = initial + age_rate * elapsed
        idx = bisect.bisect_right(priorities, prio)
        ahead = suffix_nh[idx]
        if ahead - drained <= 0:
            return DrainEstimate(rank0, len(ranked), nh0, rate, elapsed)
        drained += rate * (step / 3600.0)
        if drained >= ahead:
            return DrainEstimate(rank0, len(ranked), nh0, rate, elapsed + step)
        elapsed += step
        step = min(step * 1.15, 6 * 3600)

    return DrainEstimate(rank0, len(ranked), nh0, rate, None)


# --------------------------------------------------------------------------------------
# Estimator (a): backfill oracle
# --------------------------------------------------------------------------------------


def backfill_start(jobid: str) -> dt.datetime | None:
    text = run(["squeue", "-j", jobid, "-h", "-O", "StartTime"])
    return parse_dt(text.strip()) if text.strip() not in {"", "N/A"} else None


def backfill_testonly(spec: JobSpec) -> dt.datetime | None:
    """Ask the scheduler when this job *would* start. --test-only does not submit anything."""
    if not shutil.which("sbatch"):
        return None
    cmd = [
        "sbatch",
        "--test-only",
        f"--partition={spec.partition}",
        f"--nodes={spec.nodes}",
        f"--time={int(spec.walltime // 60)}",
        f"--qos={spec.qos}",
        f"--account={spec.account}",
        "--wrap=true",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    blob = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", blob)
    return parse_dt(match.group(1)) if match else None


# --------------------------------------------------------------------------------------
# QoS eligibility
# --------------------------------------------------------------------------------------

# Not expressible as a SLURM QoS limit (there is no MinNodes), but real: on boost_usr_prod
# bprod is reserved for large jobs. Empirically 197 jobs at exactly 65 nodes, one at 64.
QOS_MIN_NODES = {"boost_qos_bprod": 65, "dcgp_qos_bprod": 65}


def qos_violations(
    spec: JobSpec, qos_table: dict[str, QosInfo], partitions: dict[str, PartitionInfo]
) -> list[str]:
    """Reasons this spec would be rejected or is not permitted under its QoS."""
    problems: list[str] = []
    qos = qos_table.get(spec.qos)
    if qos is None:
        return [f"unknown QoS {spec.qos!r}"]

    over_part = "OverPartQOS" in qos.flags

    if qos.max_wall and spec.walltime > qos.max_wall:
        problems.append(f"walltime > QoS MaxWall ({fmt_duration(qos.max_wall)})")

    part = partitions.get(spec.partition)
    if part:
        if part.max_time and spec.walltime > part.max_time and not over_part:
            problems.append(f"walltime > partition MaxTime ({fmt_duration(part.max_time)})")
        if (
            part.max_nodes
            and spec.nodes > part.max_nodes
            and "PartitionMaxNodes" not in qos.flags
            and not over_part
        ):
            problems.append(f"nodes > partition MaxNodes ({part.max_nodes})")

    for label, table in (("MaxTRES", qos.max_tres), ("MaxTRESPU", qos.max_tres_pu)):
        limit = table.get("node")
        if limit and spec.nodes > limit:
            problems.append(f"nodes > QoS {label} node={int(limit)}")
        gpu_limit = table.get("gres/gpu")
        if gpu_limit and spec.nodes * spec.gpus_per_node > gpu_limit:
            problems.append(f"GPUs > QoS {label} gres/gpu={int(gpu_limit)}")

    grp_nodes = qos.grp_tres.get("node")
    if grp_nodes is not None and grp_nodes == 0:
        problems.append("QoS GrpTRES node=0 (closed to this cluster)")

    min_nodes = QOS_MIN_NODES.get(spec.qos)
    if min_nodes and spec.nodes < min_nodes:
        problems.append(f"nodes < {min_nodes} (site convention for {spec.qos})")

    return problems


def allowed_qos(
    user: str, account: str | None, partition: str, fairshare: dict[str, float]
) -> tuple[str, list[str]]:
    """Pick the account to use and list the QoS available to it.

    A user typically holds several associations (here: ict25/ict26 x boost/dcgp). Only the
    ones carrying this partition's QoS family can actually run here, and among those the one
    with real fairshare is the live allocation -- the others are expired or unused, and
    picking one would report a fairshare term of 0.
    """
    assocs = read_associations(user)
    if not assocs:
        return (account or "", ["normal"])

    if account:
        for acct, qoss in assocs:
            if acct == account:
                return acct, sorted(qoss)
        return account, sorted(assocs[0][1])

    family = partition.split("_")[0]  # 'boost' / 'dcgp'
    matching = [a for a in assocs if any(q.startswith(family) for q in a[1])] or assocs
    best = max(matching, key=lambda a: (fairshare.get(a[0].lower(), 0.0), len(a[1])))
    return best[0], sorted(best[1])


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


@dataclass
class Report:
    spec: JobSpec
    priority: PriorityBreakdown
    empirical: EmpiricalEstimate
    drain: DrainEstimate
    oracle: dt.datetime | None
    violations: list[str]

    def headline(self) -> float | None:
        """Single best-guess wait: the oracle if we have one, else the empirical median."""
        if self.oracle is not None:
            return max((self.oracle - dt.datetime.now()).total_seconds(), 0.0)
        if self.empirical.p50 is not None:
            return self.empirical.p50
        return self.drain.eta


def build_report(
    spec: JobSpec,
    model: PriorityModel,
    queue: QueueState,
    history: Sequence[HistoricalJob],
    qos_table: dict[str, QosInfo],
    partitions: dict[str, PartitionInfo],
    age_seconds: float = 0.0,
    oracle: dt.datetime | None = None,
) -> Report:
    priority = model.compute(spec, age_seconds=age_seconds)
    return Report(
        spec=spec,
        priority=priority,
        # age_seconds is how long this job has already been queued, so the history estimate
        # is conditional on that: for a job already waiting, the question is how much longer.
        empirical=empirical_estimate(
            history, spec, qos_table, elapsed=age_seconds, priority=priority.total
        ),
        drain=drain_estimate(queue, model, spec),
        oracle=oracle,
        violations=qos_violations(spec, qos_table, partitions),
    )


def print_report(report: Report) -> None:
    spec = report.spec
    title = f"{spec.nodes} node(s), {fmt_duration(spec.walltime)}, qos={spec.qos}"
    if spec.jobid:
        title = f"job {spec.jobid} ({spec.name}) -- " + title
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    if report.violations:
        print("\n  !! NOT SUBMITTABLE AS SPECIFIED")
        for v in report.violations:
            print(f"     - {v}")

    p = report.priority
    print(
        f"\n  priority {int(p.total):>8,}   "
        f"= qos {int(p.qos):,} + fairshare {int(p.fairshare):,} "
        f"+ age {int(p.age):,} + jobsize {int(p.jobsize):,}"
    )

    d = report.drain
    if d.contending:
        pct = 100.0 * (d.contending - d.rank) / d.contending
        print(
            f"  rank     {d.rank:>8,} of {d.contending:,} contending "
            f"(ahead of {pct:.0f}% of the queue)"
        )

    print("\n  estimated wait")
    if report.oracle is not None:
        delta = (report.oracle - dt.datetime.now()).total_seconds()
        print(
            f"    (a) backfill reservation   {fmt_duration(delta):>8}"
            f"   -> starts {report.oracle:%Y-%m-%d %H:%M}"
        )
    else:
        print("    (a) backfill reservation        n/a   (no reservation yet; too deep in queue)")

    e = report.empirical
    if e.n:
        cens = f", {e.n_censored} still waiting" if e.n_censored else ""
        label = "history p50" if not e.elapsed else "history +p50"
        print(
            f"    (b) {label:<22}{fmt_duration(e.p50):>8}"
            f"   p75 {fmt_duration(e.p75)}, p90 {fmt_duration(e.p90)}"
        )
        if e.elapsed:
            print(
                f"        ^ time still to go, given it has already waited "
                f"{fmt_duration(e.elapsed)}"
            )
        capped = f" of {e.n_raw:,} capped" if e.n_raw > e.n else ""
        print(
            f"        from n={e.n:,}{capped} across {e.n_users} users{cens} [{e.bucket}]"
        )
    else:
        print(f"    (b) history                     n/a   (no comparable jobs [{e.bucket}])")

    print(
        f"    (c) queue drain            {fmt_duration(d.eta):>8}"
        f"   {d.node_hours_ahead:,.0f} node-h ahead @ {d.drain_rate:,.0f} node-h/h"
    )
    print("        ^ strict priority order, ignores backfill -- an upper bound for small jobs")

    head = report.headline()
    print(f"\n  >> best guess: {fmt_duration(head)}")


def print_advice(
    spec: JobSpec,
    history: Sequence[HistoricalJob],
    qos_table: dict[str, QosInfo],
    partitions: dict[str, PartitionInfo],
    qos_options: Sequence[str],
    baseline_wait: float | None = None,
) -> None:
    """Rank alternative (QoS, walltime) combinations by predicted wait.

    `baseline_wait` is what staying the course costs. For a job already queued that is the
    residual wait, not a fresh submit's -- the real choice is "keep waiting another 2.5 days
    or cancel and resubmit under a faster QoS", so comparing against a fresh submit would
    badly understate the gain.
    """
    base_p50 = baseline_wait
    if base_p50 is None:
        base_p50 = empirical_estimate(history, spec, qos_table).p50

    # Walltime candidates: fractions of what was asked for, plus each QoS's own cap, so a
    # short-walltime QoS like boost_qos_dbg can surface rather than being silently excluded
    # for exceeding a limit it would meet at a shorter walltime.
    candidates: set[float] = {spec.walltime}
    candidates.update(spec.walltime * s for s in (0.75, 0.5, 0.25))
    for qos in qos_options:
        q = qos_table.get(qos)
        if q and q.max_wall and q.max_wall < spec.walltime:
            candidates.add(q.max_wall)

    # Node-count candidates. Same walltime at a different width is a different experiment,
    # so these are flagged separately -- but on this cluster the node count is often the
    # only lever that matters (the 17-64 dead zone has no fast QoS at any walltime), so
    # leaving it out would hide the one change that actually helps.
    node_candidates: set[int] = {spec.nodes}
    for qos in qos_options:
        q = qos_table.get(qos)
        if not q:
            continue
        cap = q.max_tres.get("node") or q.max_tres_pu.get("node")
        if cap:
            node_candidates.add(int(cap))
        floor = QOS_MIN_NODES.get(qos)
        if floor:
            node_candidates.add(floor)

    rows: list[tuple[float, str, int, float, EmpiricalEstimate]] = []
    for qos in qos_options:
        for nodes in sorted(node_candidates):
            for walltime in sorted(candidates):
                alt = JobSpec(
                    nodes=nodes,
                    walltime=walltime,
                    qos=qos,
                    account=spec.account,
                    partition=spec.partition,
                )
                if qos_violations(alt, qos_table, partitions):
                    continue
                est = empirical_estimate(history, alt, qos_table)
                if est.p50 is None or est.n < 10:
                    continue
                # Only worth showing if it clearly beats the current setup.
                if base_p50 is not None and est.p50 > base_p50 * 0.5:
                    continue
                rows.append((est.p50, qos, nodes, walltime, est))

    if not rows:
        return

    # Best option per (qos, node bucket) -- two node counts in the same bucket share a
    # history pool and would print identical numbers. Among ties prefer the longest
    # walltime and the fewest nodes: no reason to cut walltime or claim more of the
    # machine than actually helps.
    best: dict[tuple[str, str], tuple[float, str, int, float, EmpiricalEstimate]] = {}
    for row in sorted(rows, key=lambda r: (r[0], -r[3], r[2])):
        best.setdefault((row[1], bucket_of(row[2], NODE_BUCKETS)), row)

    print("\n  alternatives that would cut the wait")
    print(f"    {'qos':<20}{'nodes':>6}{'walltime':>10}{'p50':>9}{'p90':>9}{'n':>7}  speedup")
    for p50, qos, nodes, walltime, est in sorted(best.values(), key=lambda r: r[0])[:8]:
        speedup = ""
        if base_p50 is not None and p50 < base_p50:
            ratio = base_p50 / max(p50, 1.0)
            # Below a second the ratio is measuring rounding, not a real difference.
            speedup = ">1000x" if ratio > 1000 else f"{ratio:.0f}x"
        mark = "" if nodes == spec.nodes else "*"
        print(
            f"    {qos:<20}{str(nodes) + mark:>6}{fmt_duration(walltime):>10}"
            f"{fmt_duration(p50):>9}{fmt_duration(est.p90):>9}{est.n:>7}  {speedup}"
        )
    if any(n != spec.nodes for _, _, n, _, _ in best.values()):
        print("    * different node count -- a different run, not the same job rescheduled")


def print_rules(
    config: PriorityConfig,
    qos_table: dict[str, QosInfo],
    partitions: dict[str, PartitionInfo],
    queue: QueueState,
    user: str,
    partition: str,
) -> None:
    """Scheduler rules that materially affect wait but are easy to miss."""
    print("\n" + "=" * 78)
    print("scheduler rules worth knowing")
    print("=" * 78)

    lines: list[str] = []

    bf = config.bf_max_job_user_part
    if bf:
        mine = sum(1 for j in queue.pending if j.user == user)
        note = f"  (you have {mine} pending here)" if mine else ""
        lines.append(
            f"backfill tests only your first {bf} pending jobs per partition{note}."
        )
        if mine > bf:
            lines.append(
                f"  ^^ you are over that limit: jobs beyond #{bf} are invisible to backfill "
                "until earlier ones clear. Submit in waves instead of all at once."
            )

    part = partitions.get(partition)
    if part and part.max_nodes:
        floor = QOS_MIN_NODES.get(f"{partition.split('_')[0]}_qos_bprod")
        if floor and floor > part.max_nodes:
            lines.append(
                f"the {part.max_nodes}-node partition cap sits just below the {floor}-node "
                f"bprod floor, so mid-size jobs (roughly 17-{part.max_nodes} nodes) have no "
                "fast lane -- measurably the slowest size on this cluster. Either stay small "
                f"or go to {floor}+ nodes under bprod."
            )

    for name, q in sorted(qos_table.items(), key=lambda kv: -kv[1].priority):
        if not name.startswith(partition.split("_")[0]):
            continue
        bits = [f"prio {int(q.priority)}"]
        if q.max_wall:
            bits.append(f"max {fmt_duration(q.max_wall)}")
        nodes = q.max_tres.get("node") or q.max_tres_pu.get("node")
        if nodes:
            bits.append(f"<={int(nodes)} nodes")
        if q.max_jobs_pu:
            bits.append(f"only {q.max_jobs_pu} running/user")
        if q.max_submit_pu:
            bits.append(f"only {q.max_submit_pu} queued/user")
        lines.append(f"{name}: " + ", ".join(bits))

    if config.small_relative_to_time:
        lines.append(
            "SMALL_RELATIVE_TO_TIME: the job-size priority term is divided by walltime, so a "
            "shorter --time raises priority AND fits backfill gaps better. Ask for what you need."
        )
    if config.usage_reset_period:
        lines.append(
            f"fairshare usage resets {config.usage_reset_period.lower()} -- a big push costs "
            "least right after a reset."
        )
    deny = [n for n, q in qos_table.items() if "DenyOnLimit" in q.flags]
    if deny:
        lines.append(
            "most QoS carry DenyOnLimit: exceeding a limit rejects the job at submit time "
            "rather than queueing it, so a typo fails loudly rather than waiting forever."
        )

    for line in lines:
        if line.startswith("  "):
            print(f"    {line.strip()}")
        else:
            print(f"  - {line}")


# --------------------------------------------------------------------------------------
# Input modes
# --------------------------------------------------------------------------------------


def spec_from_jobid(jobid: str, default_partition: str) -> tuple[JobSpec, float] | None:
    """Build a spec from a queued job, returning (spec, age_seconds)."""
    text = run(
        ["squeue", "-j", jobid, "-h", "-o", "%i|%D|%l|%q|%a|%P|%j|%V"]
    )
    line = text.strip().splitlines()
    if not line:
        return None
    cols = line[0].split("|")
    if len(cols) < 8:
        return None
    try:
        nodes = int(cols[1])
    except ValueError:
        return None
    submit = parse_dt(cols[7])
    age = (dt.datetime.now() - submit).total_seconds() if submit else 0.0
    spec = JobSpec(
        nodes=nodes,
        walltime=parse_walltime(cols[2]) or 0.0,
        qos=cols[3],
        account=cols[4],
        partition=cols[5] or default_partition,
        jobid=cols[0],
        name=cols[6],
    )
    return spec, age


SBATCH_RE = re.compile(r"^#SBATCH\s+(.*)$")


def spec_from_script(path: Path, default_partition: str, fallback_account: str) -> JobSpec:
    """Parse #SBATCH directives out of a generated run script."""
    opts: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        m = SBATCH_RE.match(raw.strip())
        if not m:
            continue
        arg = m.group(1).strip()
        if arg.startswith("--"):
            key, _, val = arg[2:].partition("=")
            if not val:
                key, _, val = arg[2:].partition(" ")
            opts[key.strip()] = val.strip()
        elif arg.startswith("-"):
            key = arg[1:2]
            opts[{"N": "nodes", "t": "time", "p": "partition", "A": "account"}.get(key, key)] = (
                arg[2:].strip()
            )
    nodes = int(opts.get("nodes", "1") or 1)
    return JobSpec(
        nodes=nodes,
        walltime=parse_walltime(opts.get("time")) or 1800.0,
        qos=opts.get("qos", "normal"),
        account=opts.get("account", fallback_account),
        partition=opts.get("partition", default_partition),
        name=opts.get("job-name", path.name),
    )


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------


def self_test(model: PriorityModel, partition: str, user: str, limit: int = 40) -> int:
    """Check the simulated priority against sprio for real queued jobs."""
    print("self-test: simulated priority vs sprio")
    print("-" * 78)

    text = run(["sprio", "-p", partition, "-h", "-o", "%i|%u|%Y|%A|%F|%J|%Q"])
    rows = [line.split("|") for line in text.splitlines() if line]
    if not rows:
        print("  no sprio output; cannot self-test")
        return 1

    # Map jobid -> (nodes, walltime, qos, account) from squeue.
    info: dict[str, tuple[int, float, str, str]] = {}
    qtext = run(["squeue", "-p", partition, "-t", "PD", "-h", "-o", "%i|%D|%l|%q|%a"])
    for line in qtext.splitlines():
        cols = line.split("|")
        if len(cols) < 5:
            continue
        try:
            info[cols[0]] = (int(cols[1]), parse_walltime(cols[2]) or 0.0, cols[3], cols[4])
        except ValueError:
            continue

    mine = [r for r in rows if len(r) > 1 and r[1] == user]
    others = [r for r in rows if len(r) > 1 and r[1] != user]
    sample = mine + others[: max(0, limit - len(mine))]

    checked = failures = 0
    for cols in sample:
        if len(cols) < 7 or cols[0] not in info:
            continue
        jobid = cols[0]
        nodes, walltime, qos, account = info[jobid]
        try:
            actual = float(cols[2])
            age_term = float(cols[3])
            fs_term = float(cols[4])
            js_term = float(cols[5])
            qos_term = float(cols[6])
        except ValueError:
            continue

        spec = JobSpec(
            nodes=nodes, walltime=walltime, qos=qos, account=account, partition=partition
        )
        # Take age and fairshare from sprio (they depend on per-job/per-account state we do
        # not have for other users) and check the terms we model from scratch.
        pred = model.compute(spec)
        checked += 1

        # The job-size term is modelled from the node fraction alone; MAX_TRES can pick a
        # memory or GPU fraction instead for oddly-shaped requests, which moves it by a unit
        # or two out of a priority in the 120k-240k range. Allow that slack.
        qos_ok = abs(pred.qos - qos_term) <= 1.0
        js_ok = abs(pred.jobsize - js_term) <= max(2.0, 0.05 * max(js_term, 1.0))
        total_pred = math.floor(pred.qos + pred.jobsize + age_term + fs_term)
        total_ok = abs(total_pred - actual) <= 2.0

        if not (qos_ok and js_ok and total_ok):
            failures += 1
            if failures <= 10:
                print(
                    f"  MISMATCH {jobid}: qos {pred.qos:.0f} vs {qos_term:.0f}, "
                    f"jobsize {pred.jobsize:.0f} vs {js_term:.0f}, "
                    f"total {total_pred} vs {actual:.0f}"
                )

    print(f"\n  checked {checked} jobs, {failures} mismatches")
    if checked and failures == 0:
        print("  OK: priority model reproduces sprio")
    return 0 if failures == 0 else 1


def backtest(partition: str, days: int, qos_table: dict[str, QosInfo]) -> None:
    """Fit on the older window, score on the most recent 24h."""
    print("backtest: predict the last 24h from the preceding window")
    print("-" * 78)
    history = fetch_history(partition, days, refresh=False)
    if not history:
        print("  no history available")
        return

    # fetch_history gives waits, not absolute times, so re-read with timestamps.
    path = cache_dir() / f"hist_{partition}_{days}d.csv.gz"
    if not path.exists():
        print("  no cache to backtest against")
        return

    cutoff = dt.datetime.now() - dt.timedelta(hours=24)
    train: list[HistoricalJob] = []
    test: list[HistoricalJob] = []
    with gzip.open(path, "rt") as fh:
        for cols in csv.reader(fh, delimiter="|"):
            if len(cols) < len(HISTORY_FIELDS):
                continue
            row = dict(zip(HISTORY_FIELDS, cols))
            eligible = parse_dt(row["Eligible"]) or parse_dt(row["Submit"])
            start = parse_dt(row["Start"])
            walltime = parse_walltime(row["Timelimit"])
            if not eligible or not start or not walltime:
                continue
            try:
                nodes = int(row["NNodes"])
            except ValueError:
                continue
            wait = (start - eligible).total_seconds()
            if wait < 0:
                continue
            job = HistoricalJob(row["QOS"], nodes, walltime, wait, censored=False)
            (test if start >= cutoff else train).append(job)

    if not train or not test:
        print(f"  insufficient data (train={len(train)}, test={len(test)})")
        return

    groups: dict[tuple[str, str], list[HistoricalJob]] = defaultdict(list)
    for job in test:
        groups[(job.qos, bucket_of(job.nodes, NODE_BUCKETS))].append(job)

    print(f"    {'qos':<20}{'nodes':<8}{'n':>6}{'pred p50':>10}{'act p50':>10}{'<=p90':>8}")
    total_n = total_covered = 0
    ratios: list[float] = []
    for (qos, nb), jobs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:12]:
        if len(jobs) < 20:
            continue
        spec = JobSpec(
            nodes=jobs[0].nodes,
            walltime=jobs[0].walltime,
            qos=qos,
            account="",
            partition=partition,
        )
        est = empirical_estimate(train, spec, qos_table)
        if est.p50 is None:
            continue
        actual = sorted(j.wait for j in jobs)
        act_p50 = actual[len(actual) // 2]
        if est.p50 > 0 and act_p50 > 0:
            ratios.append(act_p50 / est.p50)
        if est.p90 is not None:
            covered = sum(1 for w in actual if w <= est.p90)
            total_n += len(actual)
            total_covered += covered
            cov = f"{100.0 * covered / len(actual):.0f}%"
        else:
            cov = "n/a"
        print(
            f"    {qos:<20}{nb:<8}{len(jobs):>6}"
            f"{fmt_duration(est.p50):>10}{fmt_duration(act_p50):>10}{cov:>8}"
        )

    if total_n:
        print(
            f"\n  calibration: {100.0 * total_covered / total_n:.0f}% of actual waits fell "
            f"within the predicted p90 (target ~90%)"
        )
    if ratios:
        ratios.sort()
        bias = ratios[len(ratios) // 2]
        print(f"  median bias: actual p50 is {bias:.1f}x the predicted p50")
        if bias > 1.3:
            print(
                "    the queue is busier now than over the training window, so treat p50 as\n"
                "    optimistic and plan against p75/p90 (which calibrate well)."
            )

    _backtest_conditional(history, partition)


def _backtest_conditional(history: Sequence[HistoricalJob], partition: str) -> None:
    """Check the conditional-on-elapsed path, which is what already-queued jobs rely on.

    A job that has waited a day is not a fresh draw from the wait distribution -- it has
    already survived into the tail. This verifies that conditioning on elapsed time predicts
    the remaining wait, rather than the unconditional median which is wrong by ~500x here.
    """
    pool = cap_per_user(
        [j for j in history if j.qos == "normal" and bucket_of(j.nodes, NODE_BUCKETS) == "1"]
    )
    if len(pool) < 200:
        return

    curve = kaplan_meier(pool)
    print("\n  conditional check (1-node normal): residual wait given time already waited")
    print(
        f"    {'waited':>8}{'pred +p50':>11}{'actual':>9}"
        f"{'pred +p90':>11}{'<=p90':>8}{'n':>8}"
    )
    for elapsed in (0, 3600, 6 * 3600, 12 * 3600, 24 * 3600):
        actual = sorted(j.wait - elapsed for j in pool if j.wait > elapsed and not j.censored)
        if len(actual) < MIN_SAMPLES:
            continue
        p50 = km_quantile(curve, 0.50, elapsed)
        p90 = km_quantile(curve, 0.90, elapsed)
        a50 = actual[len(actual) // 2]
        cov = (
            f"{100.0 * sum(1 for w in actual if w <= p90) / len(actual):.0f}%"
            if p90 is not None
            else "n/a"
        )
        print(
            f"    {fmt_duration(elapsed):>8}{fmt_duration(p50):>11}{fmt_duration(a50):>9}"
            f"{fmt_duration(p90):>11}{cov:>8}{len(actual):>8}"
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Estimate Leonardo queue wait time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--nodes", type=int, help="node count for a hypothetical job")
    ap.add_argument("--time", help="walltime for a hypothetical job, e.g. 12:00:00")
    ap.add_argument("--qos", help="QoS (default: the partition default)")
    ap.add_argument("--partition", default=DEFAULT_PARTITION)
    ap.add_argument("--account")
    ap.add_argument("--job", action="append", default=[], help="report on a queued job id")
    ap.add_argument("--script", action="append", default=[], help="parse an sbatch script")
    ap.add_argument("--mine", action="store_true", help="report on all my pending jobs")
    ap.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    ap.add_argument("--refresh", action="store_true", help="force-refresh the history cache")
    ap.add_argument("--no-advise", action="store_true")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args(argv)

    user = os.environ.get("USER", "")
    if not shutil.which("squeue"):
        print("error: SLURM commands not found (run this on a Leonardo login node)", file=sys.stderr)
        return 2

    config = read_priority_config()
    qos_table = read_qos()
    partitions = read_partitions()
    fairshare = read_fairshare(user)
    model = PriorityModel(config, qos_table, partitions, fairshare)

    if args.self_test:
        return self_test(model, args.partition, user)
    if args.backtest:
        backtest(args.partition, args.history_days, qos_table)
        return 0

    account, qos_options = allowed_qos(user, args.account, args.partition, fairshare)

    specs: list[tuple[JobSpec, float, dt.datetime | None]] = []

    for jobid in args.job:
        got = spec_from_jobid(jobid, args.partition)
        if got is None:
            print(f"warning: job {jobid} not found in the queue", file=sys.stderr)
            continue
        spec, age = got
        specs.append((spec, age, backfill_start(jobid)))

    for script in args.script:
        path = Path(script)
        if not path.exists():
            print(f"warning: {script} not found", file=sys.stderr)
            continue
        specs.append((spec_from_script(path, args.partition, account), 0.0, None))

    if args.mine:
        text = run(["squeue", "-u", user, "-t", "PD", "-h", "-o", "%i"])
        for jobid in text.split():
            got = spec_from_jobid(jobid, args.partition)
            if got:
                spec, age = got
                specs.append((spec, age, backfill_start(jobid)))

    if args.nodes or args.time:
        walltime = parse_walltime(args.time)
        if walltime is None:
            part = partitions.get(args.partition)
            walltime = part.default_time if part and part.default_time else 1800.0
        # The partition's `QoS=` field is a partition-level cap (boost_usr_prod), not
        # something a user submits under -- jobs with no --qos land in `normal`.
        qos = args.qos
        if not qos:
            part_qos = partitions.get(args.partition, PartitionInfo("")).default_qos
            qos = part_qos if part_qos in qos_options else "normal"
        spec = JobSpec(
            nodes=args.nodes or 1,
            walltime=walltime,
            qos=qos,
            account=account,
            partition=args.partition,
        )
        specs.append((spec, 0.0, backfill_testonly(spec)))

    if not specs:
        ap.error("nothing to estimate: pass --nodes/--time, --job, --script, or --mine")

    partition = specs[0][0].partition
    queue = read_queue(partition)
    history = fetch_history(partition, args.history_days, args.refresh)

    reports = [
        build_report(
            spec, model, queue, history, qos_table, partitions, age_seconds=age, oracle=oracle
        )
        for spec, age, oracle in specs
    ]

    if args.json:
        payload = [
            {
                "nodes": r.spec.nodes,
                "walltime_s": r.spec.walltime,
                "qos": r.spec.qos,
                "partition": r.spec.partition,
                "jobid": r.spec.jobid,
                "priority": r.priority.as_dict(),
                "rank": r.drain.rank,
                "contending": r.drain.contending,
                "violations": r.violations,
                "oracle_start": r.oracle.isoformat() if r.oracle else None,
                "empirical": {
                    "n": r.empirical.n,
                    "n_censored": r.empirical.n_censored,
                    "p50_s": r.empirical.p50,
                    "p75_s": r.empirical.p75,
                    "p90_s": r.empirical.p90,
                    "bucket": r.empirical.bucket,
                },
                "elapsed_s": r.empirical.elapsed,
                "drain_eta_s": r.drain.eta,
                "best_guess_s": r.headline(),
            }
            for r in reports
        ]
        json.dump(payload, sys.stdout, indent=2)
        print()
        return 0

    print(f"partition {partition}: {len(queue.contending):,} contending of {len(queue.pending):,} pending, "
          f"{len(queue.running):,} running, {queue.partition.total_nodes:,} nodes")
    reasons = Counter(j.reason for j in queue.pending if not j.contending)
    if reasons:
        blocked = ", ".join(f"{n} {r}" for r, n in reasons.most_common(4))
        print(f"  (not competing: {blocked})")
    print(f"history window: {args.history_days}d, {len(history):,} jobs")

    for report in reports:
        print_report(report)
        if not args.no_advise:
            print_advice(
                report.spec,
                history,
                qos_table,
                partitions,
                qos_options,
                baseline_wait=report.headline(),
            )

    if not args.no_rules:
        print_rules(config, qos_table, partitions, queue, user, partition)

    return 0


if __name__ == "__main__":
    sys.exit(main())
