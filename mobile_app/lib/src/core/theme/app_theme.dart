import 'package:flutter/material.dart';

class AppColors {
  static const Color background = Color(0xFF1A1A1A); // Deep Charcoal
  static const Color surface = Color(0xFF2D2D2D);
  static const Color textPrimary = Color(0xFFF5F5F5); // Soft White
  static const Color textSecondary = Color(0xFFB0B0B0);

  // Emotion Colors
  static const Color emotionAnger = Color(0xFFFF4B4B); // Dynamic Red
  static const Color emotionCalm = Color(0xFF00BCD4); // Cyan
  static const Color emotionJoy = Color(0xFFFFC107); // Amber
  static const Color emotionNeutral = Color(0xFF9E9E9E);
  static const Color emotionFear = Color(0xFF9C27B0); // Purple

  static const Color primary = Color(0xFF00BCD4);
}

class AppTheme {
  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.background,
      primaryColor: AppColors.primary,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.primary,
        surface: AppColors.surface,
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontFamily: 'Roboto',
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: AppColors.textPrimary,
        ),
        bodyLarge: TextStyle(
          fontFamily: 'Roboto',
          fontSize: 16,
          color: AppColors.textPrimary,
        ),
        bodyMedium: TextStyle(
          fontFamily: 'Roboto',
          fontSize: 14,
          color: AppColors.textSecondary,
        ),
      ),
      /* cardTheme: CardTheme(
        color: AppColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        elevation: 4,
      ), */
    );
  }
}
