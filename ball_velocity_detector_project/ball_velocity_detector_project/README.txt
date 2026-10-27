BALL VELOCITY DETECTOR - RESEARCH BUNDLE

Files
-----
app.py: Streamlit application. Keep ball_detector.pt in the same folder as app.py, or choose another weights path in the sidebar.
train_custom_ball_model.py: Optional script for fine-tuning a YOLO detector on a labelled ball dataset.

Run the application
-------------------
From the folder containing app.py and ball_detector.pt:
  .venv\Scripts\python.exe -m streamlit run app.py

The app includes finite-difference, windowed-linear-regression and image-plane projectile-fit speed estimates; optional Kalman position smoothing; selection of custom weights or YOLO11n/s/m weights; manual scale, four-point planar homography and optional lens-undistortion inputs; known-diameter scale approximation; ByteTrack selection; confidence, resolution, smoothing and jump-threshold controls; manually entered motion-blur streak/exposure experiment; descriptive spread and illustrative propagated uncertainty; optional reference-speed error reporting; and fields to record lighting and camera angle.

Important distinction
---------------------
This is an experimental research bundle, not a claim that every requested experiment is fully validated. The app currently offers model selection for separate runs rather than an automated multi-model benchmark table. ByteTrack is available through Ultralytics; a standalone SORT implementation is not included. The known-diameter scale is a rough apparent-size approximation and does not by itself recover true 3D velocity. Blur streak length is entered manually, and the fusion is a simple equal-weight baseline. Uncertainty is illustrative unless calibration, timing and localisation errors are measured and coverage is tested over repeated trials. Lighting and camera angle are recorded as experimental metadata; controlled sensitivity trials still need to be run. The custom-model training script needs a correctly labelled dataset and must be run separately.

Calibration notes
-----------------
Four-point homography maps points on a known plane; it is not a general 3D calibration method. Lens-undistortion coefficients and camera matrix should be measured for the actual camera/resolution, not guessed. Official general-purpose YOLO weights may not detect the target ball well unless the target class is compatible or the model is fine-tuned.

Suggested detector comparison procedure
---------------------------------------
Use the same test video, calibration, confidence threshold and resolution. Run each model separately (custom weights, yolo11n.pt, yolo11s.pt, yolo11m.pt), record detection count, estimated speed, processing FPS and reference-speed error where known. Do not compare models using different clips or different calibration settings.

Custom training dataset
-----------------------
Create a YOLO-format dataset with images/train, images/val, matching labels/train and labels/val, and a data.yaml containing path, train, val, and names. Example:
  path: C:/datasets/ball_dataset
  train: images/train
  val: images/val
  names:
    0: ball
Then run:
  .venv\Scripts\python.exe train_custom_ball_model.py --data C:/datasets/ball_dataset/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
Training needs sufficient labelled examples and may be slow without a compatible GPU. Keep the held-out test clips separate from training data.
