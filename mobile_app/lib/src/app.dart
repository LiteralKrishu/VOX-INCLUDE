import 'package:flutter/material.dart';
import 'features/analysis/presentation/dashboard_screen.dart';
import 'features/analysis/presentation/watch_dashboard_screen.dart';
import 'core/theme/app_theme.dart';

class VoxIncludeApp extends StatelessWidget {
  const VoxIncludeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VOX-INCLUDE',
      theme: AppTheme.darkTheme,
      home: LayoutBuilder(
        builder: (context, constraints) {
          // Heuristic: Watches typically have width < 250
          if (constraints.maxWidth < 300 && constraints.maxHeight < 300) {
            return const WatchDashboardScreen();
          }
          return const DashboardScreen();
        },
      ),
      debugShowCheckedModeBanner: false,
    );
  }
}
