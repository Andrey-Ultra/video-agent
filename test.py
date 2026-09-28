import Vision

langs = Vision.VNRecognizeTextRequest.supportedRecognitionLanguagesForTextRecognitionLevel_revision_error_(
    1, Vision.VNRecognizeTextRequestRevision3, None  # level=1 это fast
)
print(langs)