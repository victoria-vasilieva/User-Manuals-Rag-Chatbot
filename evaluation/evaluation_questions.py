EVALUATION_DATA: list[dict[str, str]] = [
  {
    "question": "How do I reset Wi-Fi on Roborock S7 Pro Ultra?",
    "model_filter": "S7 Pro Ultra",
    "ground_truth": "Open the top cover and locate the Wi-Fi indicator. Press and hold the Power and Home buttons until you hear 'Resetting WiFi'. The reset is complete when the Wi-Fi indicator flashes slowly and the robot waits for connection.",
    "source_doc": "S7_Pro_Ultra_manual.pdf",
    "answer_type": "procedure"
  },
  {
    "question": "Can I use detergent in the water tank?",
    "model_filter": "General",
    "ground_truth": "Only cleaning solutions officially recommended by Roborock may be used. Other detergents may cause corrosion or damage.",
    "source_doc": "All manuals",
    "answer_type": "warning"
  },
  {
    "question": "Which model has automatic mop washing?",
    "model_filter": "General",
    "ground_truth": "Roborock S7 Pro Ultra, S8 MaxV Ultra and Qrevo Pro support automatic mop washing.",
    "source_doc": "Multiple manuals",
    "answer_type": "comparison"
  },
  #{
  #  "question": "Does Roborock S9 exist?",
  #  "model_filter": "General",
  #  "ground_truth": "I could not find that information in the provided manuals.",
  #  "source_doc": "None",
  #  "answer_type": "hallucination_test"
  #}
]
