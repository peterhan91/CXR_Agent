"""CXR Agent tool wrappers — thin HTTP clients to model servers."""

from tools.base import BaseCXRTool
from tools.chexagent2 import (
    CheXagent2ReportTool,
    CheXagent2SRRGTool,
    CheXagent2GroundingTool,
    CheXagent2ClassifyTool,
    CheXagent2VQATool,
)
from tools.chexone import CheXOneReportTool
from tools.chexzero import CheXzeroClassifyTool
from tools.medversa import (
    MedVersaReportTool,
    MedVersaClassifyTool,
    MedVersaDetectTool,
    MedVersaSegmentTool,
    MedVersaVQATool,
)
from tools.biomedparse import BiomedParseSegmentTool
from tools.medsam import MedSAMSegmentTool
from tools.medsam3 import MedSAM3SegmentTool
from tools.factchexcker import FactCheXckerVerifyTool

__all__ = [
    "BaseCXRTool",
    # CheXagent-2 (multi-task)
    "CheXagent2ReportTool",
    "CheXagent2SRRGTool",
    "CheXagent2GroundingTool",
    "CheXagent2ClassifyTool",
    "CheXagent2VQATool",
    # CheXOne
    "CheXOneReportTool",
    # CheXzero (zero-shot classification)
    "CheXzeroClassifyTool",
    # MedVersa (multi-task) — broken, kept for reference
    "MedVersaReportTool",
    "MedVersaClassifyTool",
    "MedVersaDetectTool",
    "MedVersaSegmentTool",
    "MedVersaVQATool",
    # Segmentation
    "BiomedParseSegmentTool",
    "MedSAMSegmentTool",
    "MedSAM3SegmentTool",
    # Verification
    "FactCheXckerVerifyTool",
]
