import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dio/dio.dart'; // Needed for fetching config/blob

class AudioRecorderService {
  final AudioRecorder _audioRecorder = AudioRecorder();

  Future<bool> hasPermission() async {
    if (kIsWeb) return true; // Web handles permissions via browser API

    final status = await Permission.microphone.request();
    return status == PermissionStatus.granted;
  }

  Future<void> startRecording() async {
    final hasPerm = await hasPermission();
    if (!hasPerm) throw Exception("Microphone permission denied");

    // Standard config: 16k sample rate, mono (to match backend)
    // On mobile, we save to file. On web, we might stream (but record supports Web too)

    // On mobile, we save to file. On web, we might stream (but record supports Web too)

    String path;

    if (kIsWeb) {
      // Stream is better for web but record package handles it with in-memory or blob
      await _audioRecorder.start(
        const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000),
        path: '',
      );
      return;
    } else {
      // Mobile
      final tempDir = await getTemporaryDirectory();
      path = '${tempDir.path}/temp_audio.wav';
    }

    // Encoder: WAV/PCM is best for our backend which expects simple formats
    // AAC (m4a) is default but backend needs to handle it (we added ffmpeg/librosa support so it should be fine)
    // But let's try to use wav if possible or pcm

    await _audioRecorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: path,
    );
  }

  Future<String?> stopRecording() async {
    final path = await _audioRecorder.stop();
    if (path == null) return null;

    if (kIsWeb) {
      // Web: path is typically a blob URL (e.g., blob:http://localhost:1234/uuid)
      try {
        final dio = Dio();
        final response = await dio.get(
          path,
          options: Options(responseType: ResponseType.bytes),
        );
        // Dio bytes response comes as List<int> (actually Uint8List usually but compatible)
        final bytes = response.data;
        // Ensure bytes is List<int>
        if (bytes is List<int>) {
          return base64Encode(bytes);
        }
        return null; // Should not happen if responseType is bytes
      } catch (e) {
        print("Error processing web audio blob: $e");
        return null;
      }
    }

    final file = File(path);
    final bytes = await file.readAsBytes();
    return base64Encode(bytes);
  }

  Future<void> stop() async {
    await _audioRecorder.stop();
  }

  Future<bool> isRecording() async {
    return await _audioRecorder.isRecording();
  }

  Future<Amplitude> getAmplitude() async {
    return await _audioRecorder.getAmplitude();
  }
}
