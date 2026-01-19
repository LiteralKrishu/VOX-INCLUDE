import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:state_notifier/state_notifier.dart';
import '../../audio/data/audio_recorder_service.dart';
import '../../audio/data/transcription_service.dart';
import '../../analysis/data/analysis_repository.dart';
import '../domain/models.dart';
import '../../../core/api/api_client.dart';

// Providers
final apiClientProvider = Provider((ref) => ApiClient());

final audioRecorderServiceProvider = Provider((ref) => AudioRecorderService());
final transcriptionServiceProvider = Provider((ref) => TranscriptionService());

final analysisRepositoryProvider = Provider((ref) {
  final client = ref.watch(apiClientProvider);
  return AnalysisRepository(client);
});

// Real-time amplitude provider (for visualization)
final amplitudeProvider = StateNotifierProvider<AmplitudeNotifier, double>((
  ref,
) {
  final recorder = ref.watch(audioRecorderServiceProvider);
  return AmplitudeNotifier(recorder);
});

class AmplitudeNotifier extends StateNotifier<double> {
  final AudioRecorderService _recorder;
  Timer? _timer;
  bool _isActive = false;

  AmplitudeNotifier(this._recorder) : super(0.0);

  void start() {
    if (_isActive) return;
    _isActive = true;
    // Poll amplitude every 100ms for smooth visualization
    _timer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
      try {
        final amp = await _recorder.getAmplitude();
        // amp.current is in dB, typically -160 to 0
        // Normalize to 0.0-1.0 range
        final normalized = ((amp.current + 60) / 60).clamp(0.0, 1.0);
        if (mounted) {
          state = normalized;
        }
      } catch (e) {
        // Ignore errors during amplitude polling
      }
    });
  }

  void stop() {
    _isActive = false;
    _timer?.cancel();
    _timer = null;
    state = 0.0;
  }

  @override
  void dispose() {
    stop();
    super.dispose();
  }
}

// Recording state provider
final isRecordingProvider = StateProvider<bool>((ref) => false);

// API Status enum for progress indicator
enum ApiStatus { idle, recording, sending, processing, done, error }

// API Status provider
final apiStatusProvider = StateProvider<ApiStatus>((ref) => ApiStatus.idle);

final analysisControllerProvider =
    StateNotifierProvider<AnalysisController, AsyncValue<AnalysisResult?>>((
      ref,
    ) {
      final recorder = ref.watch(audioRecorderServiceProvider);
      final transcriber = ref.watch(transcriptionServiceProvider);
      final repo = ref.watch(analysisRepositoryProvider);
      final amplitudeNotifier = ref.read(amplitudeProvider.notifier);
      final isRecordingNotifier = ref.read(isRecordingProvider.notifier);
      final apiStatusNotifier = ref.read(apiStatusProvider.notifier);
      return AnalysisController(
        recorder,
        transcriber,
        repo,
        amplitudeNotifier,
        isRecordingNotifier,
        apiStatusNotifier,
      );
    });

// Controller
class AnalysisController extends StateNotifier<AsyncValue<AnalysisResult?>> {
  final AudioRecorderService _recorder;
  final TranscriptionService _transcriber;
  final AnalysisRepository _repository;
  final AmplitudeNotifier _amplitudeNotifier;
  final StateController<bool> _isRecordingNotifier;
  final StateController<ApiStatus> _apiStatusNotifier;

  bool _isAnalyzing = false;
  String _currentTranscript = "";

  AnalysisController(
    this._recorder,
    this._transcriber,
    this._repository,
    this._amplitudeNotifier,
    this._isRecordingNotifier,
    this._apiStatusNotifier,
  ) : super(const AsyncValue.data(null));

  Future<void> startAnalysis() async {
    if (_isAnalyzing) return;
    _isAnalyzing = true;
    _isRecordingNotifier.state = true;
    _apiStatusNotifier.state = ApiStatus.recording;

    try {
      // Start initial recording
      await _recorder.startRecording();

      // Start STT
      _currentTranscript = "";
      await _transcriber.startListening(
        onResult: (text) {
          _currentTranscript = text;
        },
      );

      // Start amplitude monitoring for real-time visualization
      _amplitudeNotifier.start();

      // Start the recursive loop
      _scheduleNextAnalysis();
    } catch (e, st) {
      state = AsyncValue.error(e, st);
      _isAnalyzing = false;
      _isRecordingNotifier.state = false;
      _apiStatusNotifier.state = ApiStatus.error;
    }
  }

  void _scheduleNextAnalysis() async {
    if (!_isAnalyzing || !mounted) return;

    // Wait for buffer duration (2s for faster response)
    await Future.delayed(const Duration(seconds: 2));

    if (!_isAnalyzing || !mounted) return;

    await _processChunk();

    // Recursively schedule next only if still analyzing
    if (_isAnalyzing && mounted) {
      _scheduleNextAnalysis();
    }
  }

  Future<void> stopAnalysis() async {
    _displayStop();
  }

  void _displayStop() {
    _isAnalyzing = false;
    _isRecordingNotifier.state = false;
    _apiStatusNotifier.state = ApiStatus.idle;
    _amplitudeNotifier.stop();
    _transcriber.stop();
    _recorder.stop();
  }

  Future<void> _processChunk() async {
    try {
      // 1. Stop and capture
      _apiStatusNotifier.state = ApiStatus.sending;
      final audioBase64 = await _recorder.stopRecording();

      // 2. Restart recording immediately (to capture next chunk)
      if (_isAnalyzing) {
        await _recorder.startRecording();
        _apiStatusNotifier.state = ApiStatus.recording;
      }

      if (audioBase64 == null) {
        print("Warning: No audio data captured");
        return;
      }

      // 3. Send to API
      _apiStatusNotifier.state = ApiStatus.processing;
      state = const AsyncValue.loading();

      // Pass captured text to backend for intent analysis
      final result = await _repository.analyzeAudio(
        audioBase64,
        text: _currentTranscript,
      );

      // Reset transcript if successfully sent (or keep accumulating?
      // Depends on UX. Let's keep it to allow context, or clear if handled.
      // For now, allow it to update with new speech.)

      // 4. Update State
      if (mounted) {
        _apiStatusNotifier.state = ApiStatus.done;
        state = AsyncValue.data(result);
      }
    } catch (e, st) {
      print("Analysis Error: $e");
      if (mounted) {
        _apiStatusNotifier.state = ApiStatus.error;
        // Show error to user for important failures
        state = AsyncValue.error(e, st);
      }
    }
  }

  @override
  void dispose() {
    stopAnalysis();
    super.dispose();
  }
}
