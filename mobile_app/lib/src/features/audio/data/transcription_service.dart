import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:permission_handler/permission_handler.dart';

class TranscriptionService {
  final SpeechToText _speech = SpeechToText();
  bool _isAvailable = false;

  Future<bool> init() async {
    if (!kIsWeb) {
      final status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) return false;
    }

    try {
      _isAvailable = await _speech.initialize(
        onError: (e) => debugPrint("STT Error: $e"),
        onStatus: (s) => debugPrint("STT Status: $s"),
      );
      return _isAvailable;
    } catch (e) {
      debugPrint("STT Init Error: $e");
      return false;
    }
  }

  Future<void> startListening({required Function(String) onResult}) async {
    if (!_isAvailable) {
      bool initialized = await init();
      if (!initialized) return;
    }

    await _speech.listen(
      onResult: (result) {
        if (result.finalResult) {
          onResult(result.recognizedWords);
        }
      },
      localeId: 'en_US',
      listenOptions: SpeechListenOptions(
        cancelOnError: true,
        listenMode: ListenMode.confirmation,
      ),
    );
  }

  Future<void> stop() async {
    await _speech.stop();
  }

  bool get isListening => _speech.isListening;
}
