import cv2
import numpy as np
import warnings
import os
import time
import threading
from collections import deque, namedtuple
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

try:
    from playsound import playsound

    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("Warning: playsound not available. Install with: pip install playsound")

# ==========================
# CONFIGURATION
# ==========================
CAMERA_INDEX = 0
DEBUG_MODE = False

# HSV Color Ranges
LOWER_GREEN = np.array([35, 70, 70])
UPPER_GREEN = np.array([90, 255, 255])
LOWER_RED1 = np.array([0, 70, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 70, 70])
UPPER_RED2 = np.array([179, 255, 255])

# Detection Thresholds
MIN_RECT_AREA = 1500
MIN_CIRCLE_AREA = 500
MIN_ASPECT_RATIO = 1.8
MAX_ASPECT_RATIO = 5.5
MIN_CIRCULARITY = 0.65
DEBOUNCE_FRAMES = 3
RED_DURATION_THRESHOLD = 5.0
STATE_TIMEOUT = 2.0
STALE_TIMEOUT = STATE_TIMEOUT * 1.5  # Allow reset state to persist

# State constants
STATE_RED = "RED"
STATE_GREEN = "GREEN"

# Matching thresholds
IOU_THRESHOLD = 0.4
MAX_DISTANCE = 100
DISTANCE_THRESHOLD = 0.6

# Visual
KERNEL = np.ones((5, 5), np.uint8)
RECT_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

# System - 3 second retry timeout
CAMERA_RETRY_TIMEOUT = 3.0
CAMERA_RETRY_DELAY = 0.1
MAX_CAMERA_RETRIES = int(CAMERA_RETRY_TIMEOUT / CAMERA_RETRY_DELAY)

# Sound
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_PATH = os.path.join(BASE_DIR, "assets", "sound.mp3")

Scope = namedtuple('Scope', ['x', 'y', 'w', 'h'])


def scope_centroid(scope: Scope) -> Tuple[float, float]:
    """Calculate scope centroid."""
    return (scope.x + scope.w / 2.0, scope.y + scope.h / 2.0)


def scope_distance(s1: Scope, s2: Scope) -> float:
    """Calculate distance between scope centroids."""
    cx1, cy1 = scope_centroid(s1)
    cx2, cy2 = scope_centroid(s2)
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def scope_iou(s1: Scope, s2: Scope) -> float:
    """Calculate Intersection over Union."""
    x1 = max(s1.x, s2.x)
    y1 = max(s1.y, s2.y)
    x2 = min(s1.x + s1.w, s2.x + s2.w)
    y2 = min(s1.y + s1.h, s2.y + s2.h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    union = s1.w * s1.h + s2.w * s2.h - intersection
    return intersection / union if union > 0 else 0.0


def match_scopes_to_previous(new_scopes: List[Scope],
                             prev_scopes: List[Tuple[str, Scope]]) -> List[Tuple[Scope, Optional[str]]]:
    """Match new scopes to previous. Prefers IoU over distance matching."""
    if not prev_scopes:
        return [(scope, None) for scope in new_scopes]

    matches = []
    used_ids = set()

    for new_scope in new_scopes:
        best_id = None
        best_iou = 0.0
        best_distance_id = None
        best_distance_score = 0.0

        # Calculate both metrics for all candidates
        for prev_id, prev_scope in prev_scopes:
            if prev_id in used_ids:
                continue

            # IoU matching
            iou = scope_iou(new_scope, prev_scope)
            if iou >= IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_id = prev_id

            # Distance matching as fallback
            dist = scope_distance(new_scope, prev_scope)
            if dist < MAX_DISTANCE:
                similarity = 1.0 - (dist / MAX_DISTANCE)
                if similarity >= DISTANCE_THRESHOLD and similarity > best_distance_score:
                    best_distance_score = similarity
                    best_distance_id = prev_id

        # Prefer IoU match over distance match
        if best_id:
            matches.append((new_scope, best_id))
            used_ids.add(best_id)
            if DEBUG_MODE:
                print(f"Matched {best_id} (iou={best_iou:.2f})")
        elif best_distance_id:
            matches.append((new_scope, best_distance_id))
            used_ids.add(best_distance_id)
            if DEBUG_MODE:
                print(f"Matched {best_distance_id} (dist={best_distance_score:.2f})")
        else:
            matches.append((new_scope, None))

    return matches


_SOUND_LOCK = threading.Lock()
_SOUND_PLAYING = False


def play_sound_async() -> bool:
    """Play sound in background if not already playing."""
    if not SOUND_AVAILABLE or not os.path.exists(SOUND_PATH):
        return False

    global _SOUND_PLAYING

    with _SOUND_LOCK:
        if _SOUND_PLAYING:
            return False
        _SOUND_PLAYING = True

    def play():
        global _SOUND_PLAYING
        try:
            playsound(SOUND_PATH)
        except Exception as e:
            print(f"Sound error: {e}")
        finally:
            with _SOUND_LOCK:
                _SOUND_PLAYING = False

    threading.Thread(target=play, daemon=True).start()
    return True


class Tracker:
    """Tracks state of a single traffic light."""

    def __init__(self, tracker_id: str):
        self.id = tracker_id
        self.history = deque(maxlen=DEBOUNCE_FRAMES)
        self.state = None
        self.red_start = None
        self.last_update = None
        self.none_count = 0

    def update(self, detected_color: Optional[str], timestamp: float):
        """Update with new detection."""
        self.last_update = timestamp

        # Handle None - clear history if we've lost tracking
        if detected_color is None:
            self.none_count += 1
            if self.none_count >= DEBOUNCE_FRAMES:
                self._reset()
            return

        # Reset none counter and add to history
        self.none_count = 0
        self.history.append(detected_color)

        # Transition on full buffer of same color
        if len(self.history) == DEBOUNCE_FRAMES:
            if all(c == detected_color for c in self.history):
                self._transition(detected_color, timestamp)

    def _transition(self, new_state: str, timestamp: float):
        """Handle state transition."""
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state

        if DEBUG_MODE:
            print(f"[{self.id}] {old_state} -> {new_state}")

        if new_state == STATE_RED:
            if self.red_start is None:
                self.red_start = timestamp
        elif new_state == STATE_GREEN:
            if old_state == STATE_RED and self.red_start:
                duration = timestamp - self.red_start
                if duration >= RED_DURATION_THRESHOLD:
                    if play_sound_async() and DEBUG_MODE:
                        print(f"[{self.id}] Sound! ({duration:.1f}s)")
            self.red_start = None

    def _reset(self):
        """Reset tracking state."""
        if DEBUG_MODE and self.state is not None:
            print(f"[{self.id}] Reset")
        self.state = None
        self.red_start = None
        self.history.clear()
        self.none_count = 0

    def is_stale(self, timestamp: float) -> bool:
        """Check if tracker should be removed."""
        if self.last_update is None:
            return True
        return (timestamp - self.last_update) > STALE_TIMEOUT

    def get_status(self, timestamp: float) -> str:
        """Get display status."""
        if self.state == STATE_RED and self.red_start:
            duration = timestamp - self.red_start
            return f"RED: {duration:.1f}s"
        elif self.state == STATE_GREEN:
            return "GREEN"
        return "..."


class Detector:
    """Traffic light detection system."""

    def __init__(self):
        self.cap = None
        self.trackers = {}
        self.prev_scopes = []
        self.next_id = 0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_time = time.time()
        self.retry_count = 0

    def init_camera(self) -> bool:
        """Initialize camera with retry logic."""
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError("Camera failed to open")
            self.retry_count = 0
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False

    def find_scopes(self, frame: np.ndarray) -> List[Scope]:
        """Find traffic light rectangular scopes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scopes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_RECT_AREA:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            ratio = h / w if w > 0 else 0

            if MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
                scopes.append(Scope(x, y, w, h))

        return scopes

    def find_circle_in_scope(self, scope: Scope, combined_mask: np.ndarray,
                             mask_green: np.ndarray, mask_red: np.ndarray) -> Tuple[Optional[str], Optional[Tuple]]:
        """Find circular light in scope region."""
        x, y, w, h = scope
        mask_height, mask_width = mask_green.shape

        # Validate bounds
        if y < 0 or x < 0 or y + h > mask_height or x + w > mask_width:
            return None, None

        # Crop to scope region
        region = combined_mask[y:y + h, x:x + w]

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_detected_color = None
        best_circle_data = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CIRCLE_AREA:
                continue

            (local_x, local_y), radius = cv2.minEnclosingCircle(cnt)

            # Check circularity
            circle_area = np.pi * radius * radius
            if circle_area == 0:
                continue
            circularity = area / circle_area
            if circularity < MIN_CIRCULARITY:
                continue

            # Convert to global coordinates
            global_x = int(local_x + x)
            global_y = int(local_y + y)

            # Bounds check
            if not (0 <= global_y < mask_height and 0 <= global_x < mask_width):
                continue

            # Determine color
            if mask_green[global_y, global_x] > 0:
                detected_color = STATE_GREEN
                draw_color = GREEN_COLOR
            elif mask_red[global_y, global_x] > 0:
                detected_color = STATE_RED
                draw_color = RED_COLOR
            else:
                continue

            if area > best_area:
                best_area = area
                best_detected_color = detected_color
                best_circle_data = (global_x, global_y, int(radius), draw_color)

        return best_detected_color, best_circle_data

    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        timestamp = time.time()
        elapsed = timestamp - self.fps_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_time = timestamp

    def cleanup_trackers(self, timestamp: float):
        """Remove stale trackers."""
        stale = [tid for tid, t in self.trackers.items() if t.is_stale(timestamp)]
        for tid in stale:
            print(f"Removed stale tracker: {tid}")
            del self.trackers[tid]

    def run(self):
        """Main loop."""
        if not self.init_camera():
            return

        print("Traffic Light Detector")
        print(f"Debug: {'ON' if DEBUG_MODE else 'OFF'}")
        print(f"Camera retry timeout: {CAMERA_RETRY_TIMEOUT}s")
        print("Press 'q' to quit\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.retry_count += 1
                    if self.retry_count >= MAX_CAMERA_RETRIES:
                        print("Camera connection lost")
                        break
                    time.sleep(CAMERA_RETRY_DELAY)
                    continue

                self.retry_count = 0
                timestamp = time.time()

                # Find scopes
                scopes = self.find_scopes(frame)

                if not scopes:
                    # No scopes - cleanup and show frame
                    self.cleanup_trackers(timestamp)
                    self.update_fps()
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_COLOR, 2)
                    cv2.imshow("Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Store frame dimensions
                frame_height, frame_width = frame.shape[:2]

                # Create color masks
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
                mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
                mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                # Combined mask with morphology
                combined_mask = cv2.bitwise_or(mask_green, mask_red)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, KERNEL)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, KERNEL)

                # Match scopes to previous frame
                matches = match_scopes_to_previous(scopes, self.prev_scopes)

                # Update tracking
                new_prev_scopes = []

                for scope, tracker_id in matches:
                    if tracker_id is None:
                        tracker_id = f"L{self.next_id}"
                        self.next_id += 1
                        if DEBUG_MODE:
                            print(f"New tracker: {tracker_id}")

                    new_prev_scopes.append((tracker_id, scope))

                    # Get or create tracker
                    if tracker_id not in self.trackers:
                        self.trackers[tracker_id] = Tracker(tracker_id)

                    tracker = self.trackers[tracker_id]

                    # Find circle
                    detected_color, circle_data = self.find_circle_in_scope(
                        scope, combined_mask, mask_green, mask_red
                    )

                    # Update tracker
                    tracker.update(detected_color, timestamp)

                    # Draw scope
                    x, y, w, h = scope
                    cv2.rectangle(frame, (x, y), (x + w, y + h), RECT_COLOR, 2)

                    # Draw status
                    status = tracker.get_status(timestamp)
                    text_y = y - 10 if y > 30 else y + h + 20
                    cv2.putText(frame, status, (x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RECT_COLOR, 2)
                    cv2.putText(frame, tracker_id, (x + 5, y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Draw circle
                    if circle_data:
                        center_x, center_y, radius, draw_color = circle_data
                        cv2.circle(frame, (center_x, center_y), radius, draw_color, 3)
                        cv2.circle(frame, (center_x, center_y), 2, (255, 255, 255), -1)

                # Update stored scopes
                self.prev_scopes = new_prev_scopes

                # Cleanup stale trackers
                self.cleanup_trackers(timestamp)

                # Update display
                self.update_fps()
                cv2.putText(frame, f"FPS: {self.fps:.1f} | Lights: {len(scopes)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_COLOR, 2)
                cv2.imshow("Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopped")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    Detector().run()