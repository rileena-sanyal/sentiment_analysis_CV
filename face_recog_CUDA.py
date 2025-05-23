import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from retinaface import RetinaFace
import torch.nn.functional as F
from collections import defaultdict
import json
from datetime import datetime
import time

# Check for emotion recognition models
try:
    from fer import FER

    USE_FER = True
except ImportError:
    USE_FER = False
    print("FER not available, using alternative emotion detection")

# Try importing facenet-pytorch for better GPU performance
try:
    from facenet_pytorch import MTCNN

    USE_FACENET = True
except ImportError:
    USE_FACENET = False


class CUDAVideoSentimentAnalyzer:
    def __init__(self, device=None):
        """Initialize CUDA-accelerated video sentiment analyzer."""
        # Auto-detect CUDA device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize models on GPU
        self.setup_models()

        # Tracking and history
        self.face_tracker = {}
        self.sentiment_history = defaultdict(list)
        self.frame_count = 0

        # Colors for different sentiments
        self.sentiment_colors = {
            'positive': (0, 255, 0),  # Green
            'negative': (0, 0, 255),  # Red
            'neutral': (255, 255, 0),  # Yellow
            'unknown': (128, 128, 128)  # Gray
        }

        # Performance tracking
        self.processing_times = []

        # Preprocessing transforms for GPU
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup_models(self):
        """Setup all models on GPU."""
        print("Setting up models on GPU...")

        # Face detection with MTCNN (GPU accelerated)
        if USE_FACENET:
            try:
                self.mtcnn = MTCNN(
                    image_size=160,
                    margin=0,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=False,
                    device=self.device
                )
                print("✓ MTCNN (GPU) initialized for face detection")
                self.use_mtcnn = True
            except Exception as e:
                print(f"MTCNN failed, falling back to RetinaFace: {e}")
                self.use_mtcnn = False
        else:
            self.use_mtcnn = False

        # Emotion detection
        if USE_FER:
            try:
                self.emotion_detector = FER(mtcnn=True)
                print("✓ FER emotion detector initialized")
                self.use_fer = True
            except Exception as e:
                print(f"FER initialization failed: {e}")
                self.use_fer = False
        else:
            self.use_fer = False

        # Create a simple emotion classifier using PyTorch (fallback)
        if not self.use_fer:
            self.setup_pytorch_emotion_model()

    def setup_pytorch_emotion_model(self):
        """Setup a simple PyTorch emotion model as fallback."""
        print("Setting up PyTorch emotion model...")

        class SimpleEmotionNet(torch.nn.Module):
            def __init__(self, num_classes=7):
                super(SimpleEmotionNet, self).__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(128 * 7 * 7, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(512, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        self.emotion_model = SimpleEmotionNet().to(self.device)
        # Initialize with random weights (in real scenario, you'd load pre-trained weights)
        self.emotion_model.eval()

        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print("✓ PyTorch emotion model initialized")

    def detect_faces_gpu(self, frame):
        """GPU-accelerated face detection."""
        if self.use_mtcnn:
            return self.detect_faces_mtcnn(frame)
        else:
            return self.detect_faces_retinaface(frame)

    def detect_faces_mtcnn(self, frame):
        """Detect faces using MTCNN (GPU)."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            boxes, probs = self.mtcnn.detect(rgb_frame)

            faces = {}
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob > 0.7:  # Confidence threshold
                        x1, y1, x2, y2 = box.astype(int)
                        faces[f'face_{i}'] = {
                            'facial_area': [x1, y1, x2, y2],
                            'score': prob
                        }
            return faces
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return {}

    def detect_faces_retinaface(self, frame):
        """Fallback to RetinaFace detection."""
        try:
            faces = RetinaFace.detect_faces(frame)
            return faces
        except:
            return {}

    def preprocess_faces_batch(self, frame, faces):
        """Preprocess multiple faces for batch processing on GPU."""
        face_tensors = []
        face_info = []

        for face_key, face_data in faces.items():
            area = face_data['facial_area']
            x1, y1, x2, y2 = area

            # Add padding
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            face_region = frame[y1:y2, x1:x2]

            if face_region.size > 0:
                try:
                    # Preprocess for GPU
                    face_tensor = self.transform(face_region)
                    face_tensors.append(face_tensor)
                    face_info.append((face_key, face_data))
                except Exception as e:
                    print(f"Preprocessing error for {face_key}: {e}")
                    continue

        if face_tensors:
            # Stack into batch tensor and move to GPU
            batch_tensor = torch.stack(face_tensors).to(self.device)
            return batch_tensor, face_info

        return None, []

    def analyze_sentiment_batch_gpu(self, face_batch):
        """Analyze sentiment for batch of faces on GPU."""
        if face_batch is None:
            return []

        results = []

        try:
            with torch.no_grad():
                if self.use_fer:
                    # Use FER for each face (CPU-based but still fast)
                    for i in range(face_batch.shape[0]):
                        face_tensor = face_batch[i]

                        # Convert back to numpy for FER
                        face_np = face_tensor.cpu().permute(1, 2, 0).numpy()
                        face_np = (face_np * 255).astype(np.uint8)

                        result = self.analyze_single_face_fer(face_np)
                        results.append(result)
                else:
                    # Use PyTorch model for batch processing
                    emotions_logits = self.emotion_model(face_batch)
                    emotions_probs = F.softmax(emotions_logits, dim=1)

                    for i in range(emotions_probs.shape[0]):
                        probs = emotions_probs[i].cpu().numpy()
                        emotion_idx = np.argmax(probs)
                        emotion = self.emotion_labels[emotion_idx]
                        confidence = probs[emotion_idx]

                        # Map to sentiment
                        sentiment_mapping = {
                            'happy': 'positive',
                            'surprise': 'positive',
                            'neutral': 'neutral',
                            'sad': 'negative',
                            'angry': 'negative',
                            'disgust': 'negative',
                            'fear': 'negative'
                        }

                        sentiment = sentiment_mapping.get(emotion, 'neutral')

                        results.append({
                            'sentiment': sentiment,
                            'emotion': emotion,
                            'confidence': float(confidence),
                            'all_emotions': {self.emotion_labels[j]: float(probs[j]) for j in range(len(probs))}
                        })

        except Exception as e:
            print(f"Batch processing error: {e}")
            # Fallback to empty results
            results = [self.get_default_result() for _ in range(face_batch.shape[0] if face_batch is not None else 0)]

        return results

    def analyze_single_face_fer(self, face_np):
        """Analyze single face with FER."""
        try:
            emotions = self.emotion_detector.detect_emotions(face_np)
            if emotions:
                emotion_scores = emotions[0]['emotions']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]

                sentiment_mapping = {
                    'happy': 'positive',
                    'surprise': 'positive',
                    'neutral': 'neutral',
                    'sad': 'negative',
                    'angry': 'negative',
                    'disgust': 'negative',
                    'fear': 'negative'
                }

                sentiment = sentiment_mapping.get(dominant_emotion, 'neutral')

                return {
                    'sentiment': sentiment,
                    'emotion': dominant_emotion,
                    'confidence': confidence,
                    'all_emotions': emotion_scores
                }
        except Exception as e:
            print(f"FER analysis error: {e}")

        return self.get_default_result()

    def get_default_result(self):
        """Get default result for failed analysis."""
        return {
            'sentiment': 'unknown',
            'emotion': 'unknown',
            'confidence': 0.0,
            'all_emotions': {}
        }

    def track_face(self, face_info, face_id):
        """Simple face tracking based on position."""
        area = face_info['facial_area']
        center_x = (area[0] + area[2]) / 2
        center_y = (area[1] + area[3]) / 2

        min_distance = float('inf')
        closest_id = None

        for existing_id, last_pos in self.face_tracker.items():
            distance = np.sqrt((center_x - last_pos[0]) ** 2 + (center_y - last_pos[1]) ** 2)
            if distance < min_distance and distance < 100:
                min_distance = distance
                closest_id = existing_id

        if closest_id:
            self.face_tracker[closest_id] = (center_x, center_y)
            return closest_id
        else:
            new_id = f"Person {len(self.face_tracker) + 1}"
            self.face_tracker[new_id] = (center_x, center_y)
            return new_id

    def draw_face_info(self, frame, face_info, person_id, sentiment_result):
        """Draw face bounding box and sentiment information on frame."""
        x1, y1, x2, y2 = face_info['facial_area']
        sentiment = sentiment_result['sentiment']
        emotion = sentiment_result['emotion']
        confidence = sentiment_result['confidence']

        color = self.sentiment_colors.get(sentiment, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare text
        texts = [
            person_id,
            f"{sentiment.upper()}",
            f"{emotion.capitalize()}",
            f"{confidence:.2f}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = y1 - 10

        for text in texts:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Background rectangle
            cv2.rectangle(frame,
                          (x1, y_offset - text_size[1] - 5),
                          (x1 + text_size[0] + 10, y_offset + 5),
                          color, -1)

            # Text
            cv2.putText(frame, text, (x1 + 5, y_offset),
                        font, font_scale, (255, 255, 255), thickness)

            y_offset -= text_size[1] + 10

        return frame

    def draw_performance_info(self, frame):
        """Draw performance statistics."""
        height, width = frame.shape[:2]

        # Calculate average processing time
        avg_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0

        stats_text = [
            f"Device: {self.device}",
            f"Frame: {self.frame_count}",
            f"Processing FPS: {fps:.1f}",
            f"Avg Time: {avg_time * 1000:.1f}ms",
            f"People: {len(self.face_tracker)}"
        ]

        # Overall sentiment distribution
        if self.sentiment_history:
            all_sentiments = []
            for history in self.sentiment_history.values():
                all_sentiments.extend([h['sentiment'] for h in history])

            if all_sentiments:
                sentiment_counts = defaultdict(int)
                for s in all_sentiments:
                    sentiment_counts[s] += 1

                total = len(all_sentiments)
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total) * 100
                    stats_text.append(f"{sentiment.capitalize()}: {percentage:.1f}%")

        # Draw statistics box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        max_width = 0
        total_height = 0
        for text in stats_text:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])
            total_height += text_size[1] + 5

        stats_x = width - max_width - 20
        stats_y = 20

        # Background
        cv2.rectangle(frame,
                      (stats_x - 10, stats_y - 10),
                      (stats_x + max_width + 10, stats_y + total_height + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(frame,
                      (stats_x - 10, stats_y - 10),
                      (stats_x + max_width + 10, stats_y + total_height + 10),
                      (255, 255, 255), 2)

        # Text
        y_pos = stats_y + 15
        for text in stats_text:
            cv2.putText(frame, text, (stats_x, y_pos),
                        font, font_scale, (255, 255, 255), thickness)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            y_pos += text_size[1] + 5

        return frame

    def process_video_stream(self, video_path=None, save_output=False, output_filename="cuda_sentiment_output.mp4"):
        """Process video with CUDA acceleration, but keep overlays visible on skipped frames."""
        if video_path is None:
            cap = cv2.VideoCapture(0)
            print("Using webcam for real-time CUDA-accelerated analysis...")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"Processing video with CUDA: {video_path}")

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if video_path else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer (if requested)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height)) if save_output else None

        print("CUDA-Accelerated Sentiment Analysis")
        print("Controls: 'q' to quit, 's' to save frame, 'r' to reset")
        print(f"Using device: {self.device}")

        # How often to actually run detection
        frame_skip = 2 if self.device.type == 'cuda' else 5
        last_overlay_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self.frame_count += 1

            # If we're skipping this frame, just show the last overlayed one
            if self.frame_count % frame_skip != 0:
                to_show = last_overlay_frame if last_overlay_frame is not None else frame
                cv2.imshow('CUDA Sentiment Analysis', to_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Otherwise do the full detect→draw→record pipeline
            start_time = time.time()
            original_frame = frame.copy()
            try:
                faces = self.detect_faces_gpu(frame)
                if faces:
                    face_batch, face_info_list = self.preprocess_faces_batch(frame, faces)
                    sentiment_results = self.analyze_sentiment_batch_gpu(face_batch)

                    for (face_key, face_info), sentiment_result in zip(face_info_list, sentiment_results):
                        person_id = self.track_face(face_info, face_key)
                        self.sentiment_history[person_id].append({
                            'frame': self.frame_count,
                            **sentiment_result
                        })
                        if len(self.sentiment_history[person_id]) > 30:
                            self.sentiment_history[person_id] = self.sentiment_history[person_id][-30:]
                        frame = self.draw_face_info(frame, face_info, person_id, sentiment_result)

                frame = self.draw_performance_info(frame)

                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]

            except Exception as e:
                print(f"Error processing frame {self.frame_count}: {e}")
                frame = original_frame

            # Show and (if desired) save this fully-annotated frame
            cv2.imshow('CUDA Sentiment Analysis', frame)
            if save_output and out:
                out.write(frame)

            # Cache it so skipped frames still look annotated
            last_overlay_frame = frame.copy()

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'cuda_sentiment_frame_{self.frame_count}.jpg', frame)
                print("Frame saved")
            elif key == ord('r'):
                self.face_tracker.clear()
                self.sentiment_history.clear()
                print("Tracking reset")

        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.print_summary()

    def print_summary(self):
        """Print performance and sentiment summary."""
        print("\n" + "=" * 50)
        print("CUDA SENTIMENT ANALYSIS SUMMARY")
        print("=" * 50)

        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            avg_fps = 1.0 / avg_time
            print(f"Average processing time: {avg_time * 1000:.1f}ms")
            print(f"Average processing FPS: {avg_fps:.1f}")
            print(f"Device used: {self.device}")

        for person_id, history in self.sentiment_history.items():
            if not history:
                continue

            print(f"\n{person_id.upper()}:")
            print("-" * 30)

            sentiments = [h['sentiment'] for h in history]
            emotions = [h['emotion'] for h in history]

            sentiment_counts = defaultdict(int)
            emotion_counts = defaultdict(int)

            for s in sentiments:
                sentiment_counts[s] += 1
            for e in emotions:
                emotion_counts[e] += 1

            print(f"Total detections: {len(history)}")

            print("\nSentiment Distribution:")
            for sentiment, count in sorted(sentiment_counts.items()):
                percentage = (count / len(history)) * 100
                print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")


def main():
    """Main function with CUDA support."""
    print("CUDA-Accelerated Sentiment Analysis")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")

    analyzer = CUDAVideoSentimentAnalyzer()

    print("\nChoose input source:")
    print("1. Webcam")
    print("2. Video file")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        analyzer.process_video_stream(video_path=None, save_output=False)
    elif choice == "2":
        video_path = input("Enter video file path: ").strip()
        save_choice = input("Save output video? (y/n): ").strip().lower()

        analyzer.process_video_stream(
            video_path=video_path,
            save_output=(save_choice == 'y'),
            output_filename="cuda_sentiment_output.mp4"
        )
    else:
        print("Invalid choice. Using webcam...")
        analyzer.process_video_stream(video_path=None)


if __name__ == "__main__":
    main()