import 'package:flutter/material.dart';

/// MeaningRibbon - Visualizes emotional state and real-time audio amplitude
///
/// The ribbon displays actual audio energy (not a fake loop) when recording.
/// Color represents emotion, amplitude shows audio level, jitter shows cognitive load.
class MeaningRibbon extends StatelessWidget {
  final Color color;
  final double amplitude; // 0.0 to 1.0 (from real audio)
  final double jitter; // 0.0 to 1.0 (cognitive load)
  final String? label;
  final String? transcript;
  final bool isRecording;

  const MeaningRibbon({
    super.key,
    this.color = Colors.cyan,
    this.amplitude = 0.0,
    this.jitter = 0.0,
    this.label,
    this.transcript,
    this.isRecording = false,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Real-time audio visualization (no looping animation)
        // Real-time Transcript / Text Visualization
        Container(
          height: 200,
          width: double.infinity,
          alignment: Alignment.center,
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            child: Text(
              (transcript != null && transcript!.isNotEmpty)
                  ? transcript!
                  : (isRecording ? "Listening..." : ""),
              key: ValueKey(transcript ?? "empty"),
              textAlign: TextAlign.center,
              style: TextStyle(
                color: color.withValues(
                  alpha: 0.9 + (amplitude * 0.1),
                ), // Pulse with voice
                fontSize: 24 + (amplitude * 4), // Subtle size pulse
                fontWeight: FontWeight.w300,
                height: 1.4,
                letterSpacing: 0.5,
                shadows: [
                  Shadow(
                    color: color.withValues(alpha: 0.5),
                    blurRadius: 10 * amplitude,
                  ),
                ],
              ),
            ),
          ),
        ),

        // Status indicator
        if (!isRecording)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.black45,
              borderRadius: BorderRadius.circular(25),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.mic_off, color: Colors.white54, size: 20),
                SizedBox(width: 8),
                Text(
                  "Tap mic to start",
                  style: TextStyle(color: Colors.white54, fontSize: 14),
                ),
              ],
            ),
          ),

        // Label overlay
        if (label != null && isRecording)
          Positioned(
            bottom: 20,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                label!,
                style: const TextStyle(color: Colors.white, fontSize: 16),
                textAlign: TextAlign.center,
              ),
            ),
          ),
      ],
    );
  }
}
