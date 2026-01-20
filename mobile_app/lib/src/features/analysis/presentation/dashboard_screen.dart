import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../intervention/presentation/meaning_ribbon.dart';
import 'analysis_controller.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/theme/accessibility_provider.dart';
import '../domain/models.dart';

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final analysisAsyncCallback = ref.watch(analysisControllerProvider);
    final controller = ref.read(analysisControllerProvider.notifier);
    final accessibilityMode = ref.watch(accessibilityProvider);

    // Real-time audio amplitude from microphone
    final realAmplitude = ref.watch(amplitudeProvider);
    final isRecording = ref.watch(isRecordingProvider);
    final apiStatus = ref.watch(apiStatusProvider);

    // Default values
    Color ribbonColor = AppColors.emotionNeutral;
    double ribbonJitter = 0.1;
    String? displayText = isRecording ? "Analyzing..." : null;
    AnalysisResult? data = analysisAsyncCallback.valueOrNull;

    // Error Handling
    if (analysisAsyncCallback.hasError) {
      displayText = "Error: ${analysisAsyncCallback.error}";
      ribbonColor = Colors.red;
      ribbonJitter = 0.5;
    }

    if (data != null && !analysisAsyncCallback.hasError) {
      if (data.emotion.label == 'anger') {
        ribbonColor = AppColors.emotionAnger;
      } else if (data.emotion.label == 'joy') {
        ribbonColor = AppColors.emotionJoy;
      } else if (data.emotion.label == 'calm') {
        ribbonColor = AppColors.emotionCalm;
      } else if (data.emotion.label == 'fear') {
        ribbonColor = AppColors.emotionFear;
      }

      ribbonJitter = 1.0 - data.cognitiveState.confidence;

      // Accessibility Adjustments
      if (accessibilityMode == AccessibilityMode.neurodiverse) {
        ribbonColor = ribbonColor.withValues(alpha: 0.5);
        ribbonJitter = 0.0;
      }

      // Output Mechanism Logic
      if (data.intervention.shouldIntervene) {
        displayText =
            data.intervention.adaptedContent ?? data.intervention.message;
      } else {
        if (accessibilityMode == AccessibilityMode.hearingImpaired) {
          displayText =
              "Emotion: ${data.emotion.label} (${(data.emotion.confidence * 100).toInt()}%)\nIntent: ${data.cognitiveState.state}\nTrend: ${data.emotion.trend}";
        } else {
          displayText =
              "${data.emotion.label.toUpperCase()} • ${data.cognitiveState.state}";
        }
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("VOX-INCLUDE"),
        actions: [
          PopupMenuButton<AccessibilityMode>(
            icon: const Icon(Icons.accessibility_new),
            onSelected: (mode) =>
                ref.read(accessibilityProvider.notifier).setMode(mode),
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: AccessibilityMode.standard,
                child: Text("Standard"),
              ),
              const PopupMenuItem(
                value: AccessibilityMode.hearingImpaired,
                child: Text("Hearing Impaired (Text+)"),
              ),
              const PopupMenuItem(
                value: AccessibilityMode.neurodiverse,
                child: Text("Neurodiverse (Calm)"),
              ),
            ],
          ),
        ],
      ),
      body: Stack(
        children: [
          // Main content
          Row(
            children: [
              // Main Visual Area
              Expanded(
                flex: 3,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Expanded(
                      child: MeaningRibbon(
                        color: ribbonColor,
                        amplitude: realAmplitude,
                        jitter: ribbonJitter,
                        label: displayText,
                        transcript: ref.watch(transcriptProvider),
                        isRecording: isRecording,
                      ),
                    ),

                    // Visual Aids Output Panel
                    if (data?.intervention.visualAids != null &&
                        data!.intervention.visualAids!.isNotEmpty)
                      Container(
                        height: 120,
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: ListView.builder(
                          scrollDirection: Axis.horizontal,
                          itemCount: data.intervention.visualAids!.length,
                          itemBuilder: (context, index) {
                            final aid = data.intervention.visualAids![index];
                            return Card(
                              child: Padding(
                                padding: const EdgeInsets.all(8.0),
                                child: Column(
                                  children: [
                                    const Icon(Icons.lightbulb_outline),
                                    const SizedBox(height: 8),
                                    Text(aid['type'] ?? 'Aid'),
                                    if (aid['content'] != null)
                                      Text(
                                        aid['content'].toString().substring(
                                          0,
                                          (aid['content'].toString().length)
                                              .clamp(0, 30),
                                        ),
                                        style: const TextStyle(fontSize: 10),
                                      ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      ),

                    // Intervention Alert
                    if (data?.intervention.shouldIntervene == true)
                      Container(
                        margin: const EdgeInsets.all(16),
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.orange.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.orange),
                        ),
                        child: Row(
                          children: [
                            const Icon(
                              Icons.warning_amber_rounded,
                              color: Colors.orange,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                data?.intervention.message ??
                                    "Intervention suggested",
                                style: const TextStyle(fontSize: 14),
                              ),
                            ),
                          ],
                        ),
                      ),

                    // Control Button
                    Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: ElevatedButton.icon(
                        onPressed: () {
                          if (isRecording) {
                            controller.stopAnalysis();
                          } else {
                            controller.startAnalysis();
                          }
                        },
                        icon: Icon(isRecording ? Icons.stop : Icons.mic),
                        label: Text(isRecording ? "Stop" : "Start Analysis"),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 32,
                            vertical: 16,
                          ),
                          backgroundColor: isRecording
                              ? Colors.red
                              : AppColors.primary,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // Side Panel with Metrics
              if (MediaQuery.of(context).size.width > 600)
                Container(
                  width: 250,
                  color: Colors.grey.shade900,
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Live Metrics",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      _buildMetric(
                        "Confidence",
                        data?.emotion.confidence ?? 0.0,
                      ),
                      _buildMetric("Arousal", data?.emotion.arousal ?? 0.0),
                      _buildMetric("Valence", data?.emotion.valence ?? 0.0),
                      const Divider(),
                      _buildMetric(
                        "Cog. Confidence",
                        data?.cognitiveState.confidence ?? 0.0,
                      ),
                      _buildMetric("Momentum", data?.emotion.momentum ?? 0.0),
                      _buildMetric("SNR", data?.snr ?? 0.0),
                      const Spacer(),
                      _buildMetric(
                        "Bayesian Conf.",
                        data?.cognitiveState.bayesianConfidence ?? 0.0,
                      ),
                    ],
                  ),
                ),
            ],
          ),

          // API Status Indicator in top-right corner
          Positioned(
            top: 8,
            right: 8,
            child: _buildApiStatusIndicator(apiStatus),
          ),
        ],
      ),
    );
  }

  Widget _buildApiStatusIndicator(ApiStatus status) {
    IconData icon;
    Color color;
    String label;

    switch (status) {
      case ApiStatus.idle:
        icon = Icons.circle_outlined;
        color = Colors.grey;
        label = "Idle";
        break;
      case ApiStatus.recording:
        icon = Icons.mic;
        color = Colors.green;
        label = "Recording";
        break;
      case ApiStatus.sending:
        icon = Icons.upload;
        color = Colors.orange;
        label = "Sending";
        break;
      case ApiStatus.processing:
        icon = Icons.hourglass_top;
        color = Colors.blue;
        label = "Processing";
        break;
      case ApiStatus.done:
        icon = Icons.check_circle;
        color = Colors.green;
        label = "Done";
        break;
      case ApiStatus.error:
        icon = Icons.error;
        color = Colors.red;
        label = "Error";
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (status == ApiStatus.processing || status == ApiStatus.sending)
            SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            )
          else
            Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetric(String label, double value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: AppColors.textSecondary)),
          Text(
            value.toStringAsFixed(2),
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}
