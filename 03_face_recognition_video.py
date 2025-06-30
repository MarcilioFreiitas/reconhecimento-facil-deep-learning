import os
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pickle
import numpy as np
import insightface
from numpy.linalg import norm
from ultralytics import YOLO
import concurrent.futures
import time

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    epsilon = 1e-8
    return np.dot(a, b) / (norm(a) * norm(b) + epsilon)

class UnifiedRecognizerParallel:
    """
    Sistema unificado para:
      • Reconhecimento facial com InsightFace
      • Detecção corporal via YOLOv8
      • Rastreamento assíncrono
      • Detecção de quedas via heurística de bounding‐box
    """

    def __init__(
        self,
        video_path: str,
        face_db_path: str = "trainer/face_db.pickle",
        min_similarity: float = 0.5,
        face_detection_interval: int = 10,
        scale_factor: float = 0.15,
    ):
        self.video_path = video_path
        self.min_similarity = min_similarity
        self.face_detection_interval = face_detection_interval
        self.scale_factor = scale_factor
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Carrega banco de dados facial
        if os.path.exists(face_db_path):
            with open(face_db_path, "rb") as f:
                self.face_db = pickle.load(f)
            print("[INFO] Banco de dados facial carregado.")
        else:
            self.face_db = {}
            print("[ALERTA] Banco de dados facial não encontrado!")

        # InsightFace
        self.recognizer = insightface.app.FaceAnalysis()
        self.recognizer.prepare(ctx_id=0)

        # YOLOv8 para corpo
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_conf_threshold = 0.5

        # Rastreamento facial
        self.face_tracks = []
        self.face_match_threshold = 50
        self.face_max_miss = 3
        self.lock_threshold = 0.6

        # Rastreamento corporal
        self.body_multi_tracker = cv2.legacy.MultiTracker_create()
        self.prev_body_boxes = []

        # Parâmetros de detecção de queda
        self.fall_history = 10                  # frames de histórico
        self.aspect_ratio_threshold = 0.6       # h/w menor = deitado
        self.height_drop_threshold = 0.3        # queda >30% da altura inicial
        self.fall_consecutive_frames = 3        # quadros consecutivos para disparar
        self.body_tracks = []                   # histórico de cada corpo

    def get_label_from_embedding(self, embedding: np.ndarray):
        best_label = "Desconhecido"
        best_sim = -1.0
        for label, db_emb in self.face_db.items():
            sim = cosine_similarity(embedding, db_emb)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        if best_sim >= self.min_similarity:
            return best_label, best_sim
        return "Desconhecido", best_sim

    def process_face_detection_task(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.recognizer.get(rgb)
        if faces:
            self.update_face_tracks_with_detections(faces, frame)

    def process_body_detection_task(self, frame):
        results = self.yolo_model(frame)[0]
        multi = cv2.legacy.MultiTracker_create()
        boxes = []
        for box in results.boxes:
            if int(box.cls[0]) != 0:  # classe 0 = pessoa
                continue
            conf = float(box.conf[0])
            if conf < self.yolo_conf_threshold:
                continue
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords.astype(int)
            tracker_box = (x1, y1, x2 - x1, y2 - y1)
            trk = cv2.legacy.TrackerCSRT_create()
            multi.add(trk, frame, tracker_box)
            boxes.append(tracker_box)
        return multi, boxes

    def update_face_tracks_with_detections(self, detections, frame):
        updated = [False] * len(self.face_tracks)
        for face in detections:
            x1, y1, x2, y2 = face.bbox.astype(int)
            center = ((x1 + x2)//2, (y1 + y2)//2)
            box = (x1, y1, x2-x1, y2-y1)
            label, sim = self.get_label_from_embedding(face.embedding)
            locked = (sim >= self.lock_threshold)

            matched = False
            for i, tr in enumerate(self.face_tracks):
                tx, ty = tr["center"]
                if np.hypot(center[0]-tx, center[1]-ty) < self.face_match_threshold:
                    if not tr.get("locked", False):
                        new_trk = cv2.legacy.TrackerCSRT_create()
                        new_trk.init(frame, box)
                        tr.update({
                            "bbox": (x1, y1, x2, y2),
                            "center": center,
                            "label": label,
                            "similarity": sim,
                            "locked": locked,
                            "tracker": new_trk,
                            "miss": 0
                        })
                    updated[i] = True
                    matched = True
                    break

            if not matched:
                trk = cv2.legacy.TrackerCSRT_create()
                trk.init(frame, box)
                self.face_tracks.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "label": label,
                    "similarity": sim,
                    "locked": locked,
                    "miss": 0,
                    "tracker": trk
                })
                updated.append(True)

        # incrementa miss e filtra
        for i, tr in enumerate(self.face_tracks):
            if not updated[i]:
                tr["miss"] += 1
        self.face_tracks = [
            tr for tr in self.face_tracks if tr["miss"] <= self.face_max_miss
        ]

    def update_face_tracks_with_tracking(self, frame):
        to_remove = []
        for i, tr in enumerate(self.face_tracks):
            ok, bbox = tr["tracker"].update(frame)
            if ok:
                x, y, w, h = map(int, bbox)
                tr["bbox"] = (x, y, x+w, y+h)
                tr["center"] = ((2*x+w)//2, (2*y+h)//2)
            else:
                tr["miss"] += 1
            if tr["miss"] > self.face_max_miss:
                to_remove.append(i)
        for i in reversed(to_remove):
            del self.face_tracks[i]

    def detect_fall(self, history: deque) -> bool:
        if len(history) < self.fall_history:
            return False
        x0, y0, w0, h0 = history[0]
        init_cy = y0 + h0/2
        recent = list(history)[-self.fall_consecutive_frames:]
        count = 0
        for x, y, w, h in recent:
            cy = y + h/2
            aspect = h / (w + 1e-6)
            drop = (cy - init_cy) > (h0 * self.height_drop_threshold)
            laid = aspect < self.aspect_ratio_threshold
            if drop and laid:
                count += 1
        return count >= self.fall_consecutive_frames

    def update_body_tracks(self, boxes):
        # garante tracks paralelos
        while len(self.body_tracks) < len(boxes):
            self.body_tracks.append({
                "history": deque(maxlen=self.fall_history),
                "fallen": False
            })
        # atualiza e detecta
        for i, box in enumerate(boxes):
            tr = self.body_tracks[i]
            tr["history"].append(box)
            if not tr["fallen"] and self.detect_fall(tr["history"]):
                tr["fallen"] = True
        # trim excess
        if len(self.body_tracks) > len(boxes):
            self.body_tracks = self.body_tracks[:len(boxes)]

    def process(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("[ERRO] Não foi possível abrir o vídeo.")
            return

        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w1 = int(w0 * self.scale_factor)
        h1 = int(h0 * self.scale_factor)
        print(f"[INFO] Resolução de trabalho: {w1}x{h1}")

        frame_count = 0
        body_future = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_face = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Fim do vídeo ou erro.")
                    break
                frame = cv2.resize(frame, (w1, h1))
                frame_count += 1

                # face detection assíncrona
                if frame_count % self.face_detection_interval == 0:
                    future_face = executor.submit(
                        self.process_face_detection_task, frame.copy()
                    )
                else:
                    self.update_face_tracks_with_tracking(frame)

                # body detection assíncrona
                if body_future is None or body_future.done():
                    if body_future:
                        try:
                            multi, boxes0 = body_future.result()
                            self.body_multi_tracker = multi
                            self.prev_body_boxes = boxes0
                        except Exception as e:
                            print(f"[ERRO] detect_body: {e}")
                    body_future = executor.submit(
                        self.process_body_detection_task, frame.copy()
                    )

                # update tracks
                ok, tracked_boxes = self.body_multi_tracker.update(frame)
                new_boxes = []
                if ok:
                    for idx, box in enumerate(tracked_boxes):
                        x, y, w, h = map(int, box)
                        # suaviza
                        if idx < len(self.prev_body_boxes):
                            px, py, pw, ph = self.prev_body_boxes[idx]
                            x = int(0.3*x + 0.7*px)
                            y = int(0.3*y + 0.7*py)
                            w = int(0.3*w + 0.7*pw)
                            h = int(0.3*h + 0.7*ph)
                        new_boxes.append((x, y, w, h))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 2)
                        cv2.putText(frame, "Body", (x, y-10),
                                    self.font, 0.7, (0,255,0), 2)

                # detecção de queda
                self.update_body_tracks(new_boxes)
                for i, (x, y, w, h) in enumerate(new_boxes):
                    if self.body_tracks[i]["fallen"]:
                        cv2.putText(frame, "QUEDA!", (x, y+h+25),
                                    self.font, 0.9, (0,0,255), 2)

                self.prev_body_boxes = new_boxes

                # desenha tracks faciais
                for tr in self.face_tracks:
                    x1, y1, x2, y2 = tr["bbox"]
                    color = (0,255,0) if tr.get("locked") else (0,255,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{tr['label']}:{tr['similarity']:.2f}",
                        (x2+10, y1 + (y2-y1)//2),
                        self.font, 0.8, color, 2
                    )

                cv2.imshow("Unificado: Face, Corpo e Quedas", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

def main() -> None:
    video_path = input("Caminho do vídeo (ou '0' para webcam): ").strip()
    if video_path == "0":
        video_path = 0
    recognizer = UnifiedRecognizerParallel(
        video_path=video_path,
        face_db_path="trainer/face_db.pickle",
        min_similarity=0.5,
        face_detection_interval=10,
        scale_factor=1.2
    )
    recognizer.process()

if __name__ == "__main__":
    main()
