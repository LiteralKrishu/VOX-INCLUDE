import 'package:flutter_riverpod/flutter_riverpod.dart';

enum AccessibilityMode {
  standard,
  hearingImpaired, // Emphasizes text, visual cues for audio events
  neurodiverse, // Reduced motion, calmer colors, simplified UI
}

class AccessibilityNotifier extends StateNotifier<AccessibilityMode> {
  AccessibilityNotifier() : super(AccessibilityMode.standard);

  void setMode(AccessibilityMode mode) {
    state = mode;
  }
}

final accessibilityProvider =
    StateNotifierProvider<AccessibilityNotifier, AccessibilityMode>((ref) {
      return AccessibilityNotifier();
    });
