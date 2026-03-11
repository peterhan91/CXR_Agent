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
from tools.medversa import (
    MedVersaReportTool,
    MedVersaClassifyTool,
    MedVersaDetectTool,
    MedVersaSegmentTool,
    MedVersaVQATool,
)
from tools.biomedparse import BiomedParseSegmentTool
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
    # MedVersa (multi-task)
    "MedVersaReportTool",
    "MedVersaClassifyTool",
    "MedVersaDetectTool",
    "MedVersaSegmentTool",
    "MedVersaVQATool",
    # Segmentation
    "BiomedParseSegmentTool",
    "MedSAM3SegmentTool",
    # Verification
    "FactCheXckerVerifyTool",
]
