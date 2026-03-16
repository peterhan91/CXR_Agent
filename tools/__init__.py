"""CXR Agent tool wrappers — thin HTTP clients to model servers + local tools."""

from tools.base import BaseCXRTool
from tools.evidence_board import EvidenceBoardTool
from tools.chexagent2 import (
    CheXagent2ReportTool,
    CheXagent2SRRGTool,
    CheXagent2GroundingTool,
    CheXagent2ClassifyTool,
    CheXagent2VQATool,
)
from tools.chexone import CheXOneReportTool
from tools.chexzero import CheXzeroClassifyTool
from tools.cxr_foundation import CXRFoundationClassifyTool
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
from tools.medgemma import MedGemmaVQATool, MedGemmaReportTool

__all__ = [
    "BaseCXRTool",
    # Local tools (no server)
    "EvidenceBoardTool",
    # CheXagent-2 (multi-task)
    "CheXagent2ReportTool",
    "CheXagent2SRRGTool",
    "CheXagent2GroundingTool",
    "CheXagent2ClassifyTool",
    "CheXagent2VQATool",
    # CheXOne
    "CheXOneReportTool",
    # Zero-shot classification
    "CheXzeroClassifyTool",
    "CXRFoundationClassifyTool",
    # MedGemma (VQA + report)
    "MedGemmaVQATool",
    "MedGemmaReportTool",
    # MedVersa (multi-task)
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
