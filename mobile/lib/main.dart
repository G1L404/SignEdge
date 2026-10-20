import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'src/ui/home_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // O pipeline assume o aparelho em retrato (rotação do sensor fixa).
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const SignEdgeApp());
}

class SignEdgeApp extends StatelessWidget {
  const SignEdgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SignEdge',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF14B8A6),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}
