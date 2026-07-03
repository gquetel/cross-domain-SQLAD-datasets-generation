"""MITRE ATT&CK technique vocabulary for labeling generated queries.

Single source of truth for the techniques a generated query can be tagged with,
and the ``ATT&CK label`` column that carries them. Attacks are labeled *where they
are defined*: insider objectives declare their technique alongside the sqlmap flag
or metasploit module they run (see ``ithreat_generator``), and SQLIA rows derive
theirs from their attack stage. Nothing is reverse-engineered from an identifier
after the fact, so adding or reordering an attack objective cannot silently
mislabel a dataset.

Labeling happens inline during generation, so datasets ship pre-labeled with no
separate post-processing pass required.
"""

# Discovery vs. exfiltration techniques.
T1082 = "T1082"  # System Information Discovery
T1020 = "T1020"  # Automated Exfiltration

# Name of the column carrying the per-row technique in generated datasets.
ATTACK_LABEL_COLUMN = "ATT&CK label"

# SQLIA attacks are labeled from their stage: reconnaissance enumerates structure
# (discovery); exploitation reads and extracts stored values (exfiltration).
_STAGE_TECHNIQUES = {
    "recon": T1082,
    "exploit": T1020,
}


def technique_for_stage(stage: str) -> str:
    """ATT&CK technique for a SQLIA ``attack_stage`` ("" if the stage is unknown)."""
    return _STAGE_TECHNIQUES.get(stage, "")
