# The Man in the Crowd: 1080p render farm
Pushing to `main` renders the film on GitHub Actions: 20 jobs, each renders ~500 frames at 1920x1080 (headless Chromium + SwiftShader),
grades, captions and encodes them (x264 CRF 17), and pushes the parts to branch `out-K`. Audio is mixed and the parts are joined
without re-encoding afterwards. Assets: CC0 (Poly Haven, MakeHuman, Quaternius UAL).
