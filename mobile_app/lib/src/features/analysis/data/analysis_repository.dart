import '../../../core/api/api_client.dart';
import '../domain/models.dart';

class AnalysisRepository {
  final ApiClient _apiClient;

  AnalysisRepository(this._apiClient);

  Future<AnalysisResult> analyzeAudio(
    String audioBase64, {
    String? text,
  }) async {
    // Current MVP uses the comprehensive analysis endpoint for full context
    // because it fuses emotion + intent + cognitive state
    final response = await _apiClient.post(
      '/api/v1/comprehensive-analysis',
      data: {
        'audio_base64': audioBase64,
        'sample_rate': 16000,
        'text': text, // Send transcribed text for intent analysis
      },
    );

    return AnalysisResult.fromJson(response.data);
  }
}
