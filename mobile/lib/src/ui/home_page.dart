import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../app_controller.dart';
import 'hand_overlay.dart';
import 'panels.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late final Future<AppController> _app = AppController.create();

  @override
  void dispose() {
    _app.then((app) => app.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: FutureBuilder<AppController>(
          future: _app,
          builder: (context, snapshot) {
            if (snapshot.hasError) {
              return Center(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text(
                    'Não foi possível abrir a câmera ou os modelos:\n${snapshot.error}',
                  ),
                ),
              );
            }
            final app = snapshot.data;
            if (app == null) {
              return const Center(child: CircularProgressIndicator());
            }
            return ListenableBuilder(
              listenable: app,
              builder: (context, _) => Column(
                children: [
                  TelemetryBar(app: app),
                  Expanded(child: _CameraView(app: app)),
                  ResultCard(app: app),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

class _CameraView extends StatelessWidget {
  const _CameraView({required this.app});

  final AppController app;

  @override
  Widget build(BuildContext context) {
    final camera = app.camera;
    if (camera == null || !camera.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    // previewSize vem na orientação do sensor (paisagem); em retrato, inverte.
    final preview = camera.value.previewSize!;
    final aspect = preview.height / preview.width;
    return Center(
      child: AspectRatio(
        aspectRatio: aspect,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Stack(
            fit: StackFit.expand,
            children: [
              // Camadas próprias: o overlay muda a cada frame, o preview (textura
              // nativa) e o resto da tela não precisam ser redesenhados junto.
              RepaintBoundary(child: CameraPreview(camera)),
              RepaintBoundary(
                child: CustomPaint(
                  painter: HandOverlay(
                    frameWidth: app.frameWidth,
                    frameHeight: app.frameHeight,
                    landmarks: app.landmarks,
                    view: app.view,
                    mirror: app.isFront,
                    accent: Theme.of(context).colorScheme.primary,
                  ),
                ),
              ),
              if (app.error != null)
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Container(
                    width: double.infinity,
                    color: Theme.of(context).colorScheme.errorContainer,
                    padding: const EdgeInsets.all(8),
                    child: Text(
                      app.error!,
                      maxLines: 4,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 11,
                        color: Theme.of(context).colorScheme.onErrorContainer,
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
