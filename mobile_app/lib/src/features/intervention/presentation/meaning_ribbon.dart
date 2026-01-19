import 'dart:math';
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
  final bool isRecording;

  const MeaningRibbon({
    super.key,
    this.color = Colors.cyan,
    this.amplitude = 0.0,
    this.jitter = 0.0,
    this.label,
    this.isRecording = false,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Real-time audio visualization (no looping animation)
        CustomPaint(
          size: const Size(double.infinity, 200),
          painter: AudioWavePainter(
            color: color.withValues(alpha: 0.7),
            amplitude: amplitude,
            jitter: jitter,
            isActive: isRecording,
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

/// Painter that visualizes REAL audio amplitude (not a fake animation)
class AudioWavePainter extends CustomPainter {
  final Color color;
  final double amplitude; // Real normalized amplitude 0.0-1.0
  final double jitter;
  final bool isActive;

  AudioWavePainter({
    required this.color,
    required this.amplitude,
    required this.jitter,
    required this.isActive,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()
      ..color = color.withValues(alpha: 0.2)
      ..style = PaintingStyle.fill;

    final width = size.width;
    final height = size.height;
    final midHeight = height / 2;

    if (!isActive || amplitude < 0.01) {
      // Draw flat line when not recording or silent
      canvas.drawLine(
        Offset(0, midHeight),
        Offset(width, midHeight),
        paint..color = color.withValues(alpha: 0.3),
      );
      return;
    }

    // Real audio visualization - bars representing amplitude
    final barCount = 50;
    final barWidth = width / (barCount * 1.5);
    final gap = barWidth * 0.5;

    final random = Random(42); // Fixed seed for consistent but varied heights

    for (int i = 0; i < barCount; i++) {
      final x = (i * (barWidth + gap)) + gap;

      // Create a natural-looking distribution based on real amplitude
      // Center bars are taller, edges are shorter
      final centerFactor = 1.0 - ((i - barCount / 2).abs() / (barCount / 2));
      final variation = 0.3 + random.nextDouble() * 0.7;

      // Jitter adds noise/irregularity
      final jitterNoise = jitter * (random.nextDouble() - 0.5) * 0.5;

      // Calculate bar height based on REAL amplitude
      final barHeight =
          (amplitude * centerFactor * variation + jitterNoise).clamp(
            0.05,
            1.0,
          ) *
          (height * 0.4);

      final rect = RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(x, midHeight),
          width: barWidth,
          height: barHeight,
        ),
        const Radius.circular(2),
      );

      canvas.drawRRect(rect, fillPaint);
      canvas.drawRRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant AudioWavePainter oldDelegate) {
    // Only repaint when actual values change
    return oldDelegate.amplitude != amplitude ||
        oldDelegate.color != color ||
        oldDelegate.jitter != jitter ||
        oldDelegate.isActive != isActive;
  }
}
