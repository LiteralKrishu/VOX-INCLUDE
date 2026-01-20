class Emotion {
  final String label;
  final double confidence;
  final double arousal;
  final double valence;
  final double momentum;
  final String trend;

  Emotion({
    required this.label,
    required this.confidence,
    required this.arousal,
    required this.valence,
    this.momentum = 0.0,
    this.trend = "stable",
  });

  factory Emotion.fromJson(Map<String, dynamic> json) {
    return Emotion(
      label: json['emotion'] ?? 'neutral',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      arousal: (json['arousal'] ?? 0.0).toDouble(),
      valence: (json['valence'] ?? 0.0).toDouble(),
      momentum: (json['momentum'] ?? 0.0).toDouble(),
      trend: json['trend'] ?? 'stable',
    );
  }
}

class CognitiveState {
  final String state;
  final double confidence;
  final double bayesianConfidence;
  final List<String> recommendations;

  CognitiveState({
    required this.state,
    required this.confidence,
    this.bayesianConfidence = 0.0,
    this.recommendations = const [],
  });

  factory CognitiveState.fromJson(Map<String, dynamic> json) {
    return CognitiveState(
      state: json['state'] ?? 'neutral',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      bayesianConfidence: (json['bayesian_confidence'] ?? 0.0).toDouble(),
      recommendations: List<String>.from(json['recommendations'] ?? []),
    );
  }
}

class Intervention {
  final String type;
  final String message;
  final bool shouldIntervene;
  final String? adaptedContent;
  final List<Map<String, dynamic>>? visualAids;

  Intervention({
    required this.type,
    required this.message,
    required this.shouldIntervene,
    this.adaptedContent,
    this.visualAids,
  });

  factory Intervention.fromJson(Map<String, dynamic> json) {
    return Intervention(
      type: json['intervention_type'] ?? 'none',
      message: json['message'] ?? '',
      shouldIntervene: json['should_intervene'] ?? false,
      adaptedContent: json['adapted_content'],
      visualAids: json['visual_aids'] != null
          ? List<Map<String, dynamic>>.from(json['visual_aids'])
          : null,
    );
  }
}

class AnalysisResult {
  final Emotion emotion;
  final CognitiveState cognitiveState;
  final Intervention intervention;
  final String? intent;
  final double snr;

  AnalysisResult({
    required this.emotion,
    required this.cognitiveState,
    required this.intervention,
    this.intent,
    this.snr = 0.0,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      emotion: Emotion.fromJson(json['emotion'] ?? {}),
      cognitiveState: CognitiveState.fromJson(json['cognitive_state'] ?? {}),
      intervention: Intervention.fromJson(json['intervention'] ?? {}),
      intent: json['intent']?['intent'],
      snr: (json['features']?['snr'] ?? 0.0).toDouble(),
    );
  }
}
