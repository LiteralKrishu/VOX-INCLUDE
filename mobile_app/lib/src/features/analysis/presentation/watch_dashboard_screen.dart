import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:wear/wear.dart';
import 'package:animate_do/animate_do.dart';
import 'analysis_controller.dart';
import '../../intervention/presentation/meaning_ribbon.dart';
import '../../../core/theme/app_theme.dart';

class WatchDashboardScreen extends ConsumerWidget {
  const WatchDashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return WatchShape(
      builder: (context, shape, child) {
        return _buildBody(context, ref, shape);
      },
    );
  }

  Widget _buildBody(BuildContext context, WidgetRef ref, WearShape shape) {
    final analysisAsyncCallback = ref.watch(analysisControllerProvider);
    final controller = ref.read(analysisControllerProvider.notifier);
    final isRecording = ref.watch(isRecordingProvider);
    final realAmplitude = ref.watch(amplitudeProvider);
    final apiStatus = ref.watch(apiStatusProvider);

    final data = analysisAsyncCallback.valueOrNull;
    Color statusColor = Colors.grey;
    String statusText = "Idle";

    // Status visual mapping
    if (isRecording) {
      statusColor = AppColors.primary;
      statusText = "Listening";
    }

    if (data != null) {
      if (data.emotion.label == 'anger') statusColor = AppColors.emotionAnger;
      if (data.emotion.label == 'joy') statusColor = AppColors.emotionJoy;
      if (data.emotion.label == 'calm') statusColor = AppColors.emotionCalm;
      if (data.emotion.label == 'fear') statusColor = AppColors.emotionFear;
      statusText = data.emotion.label.toUpperCase();
    }

    final isRound = shape == WearShape.round;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Container(
          width: isRound ? MediaQuery.of(context).size.width * 0.9 : null,
          height: isRound ? MediaQuery.of(context).size.height * 0.9 : null,
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Minimal Header
              Text(
                "VOX",
                style: TextStyle(
                  color: Colors.white.withOpacity(0.5),
                  fontSize: 10,
                  letterSpacing: 2,
                ),
              ),
              const SizedBox(height: 8),

              // Dynamic Status Circle
              Expanded(
                child: GestureDetector(
                  onTap: () {
                    if (isRecording) {
                      controller.stopAnalysis();
                    } else {
                      controller.startAnalysis();
                    }
                  },
                  child: Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: statusColor.withOpacity(0.2),
                      border: Border.all(
                        color: statusColor.withOpacity(0.8),
                        width: isRecording ? 2 + (realAmplitude * 5) : 2,
                      ),
                    ),
                    child: Center(
                      child: isRecording
                          ? (apiStatus == ApiStatus.processing
                                ? const CircularProgressIndicator(
                                    color: Colors.white,
                                  )
                                : Icon(
                                    Icons.stop,
                                    size: 32,
                                    color: statusColor,
                                  ))
                          : Icon(Icons.mic, size: 32, color: statusColor),
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 8),

              // Emotion Label
              if (isRecording)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: FadeIn(
                    duration: const Duration(milliseconds: 300),
                    child: Text(
                      statusText,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: statusColor,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                )
              else
                const Text(
                  "Tap to Start",
                  style: TextStyle(color: Colors.white70, fontSize: 12),
                ),

              // Intent (if available and short)
              if (data?.cognitiveState.intent != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    data!.cognitiveState.intent!,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: Colors.white54, fontSize: 10),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
