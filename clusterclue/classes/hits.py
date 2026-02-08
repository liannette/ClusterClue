from dataclasses import dataclass

@dataclass
class PrestoHit:
    bgc_id: str
    motif_id: str
    score: float
    tokenized_genes: set


# TODO: class inheritance from a base class Motif
@dataclass
class MotifHit:
    bgc_id: str
    motif_id: str
    score: float
    tokenized_genes: set