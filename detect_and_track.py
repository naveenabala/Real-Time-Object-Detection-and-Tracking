%%writefile detect_and_track.py
import argparse
import json
import time

import cv2
from ultralytics import YOLO

import config
from utils import FPSMeter, LineCounter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=config.VIDEO_SOURCE)
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    p.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    p.add_argument("--output", default=config.OUTPUT_VIDEO_PATH)
    return p.parse_args([])  # parse_args([]) so this also runs cleanly as a notebook cell


def main():
    args = parse_args()

    model = YOLO(args.model)
    class_names = model.names

    try:
        source = int(args.source)
    except (TypeError, ValueError):
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, src_fps, (width, height))

    line_counter = LineCounter(config.LINE_START, config.LINE_END)
    fps_meter = FPSMeter()
    class_detection_counts = {}
    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        fps_meter.tick()

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            classes=config.CLASSES_OF_INTEREST,
            verbose=False,
        )[0]

        annotated = frame.copy()
        cv2.line(annotated, config.LINE_START, config.LINE_END, (0, 200, 255), 2)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                cls_name = class_names.get(int(cls_id), str(cls_id))
                class_detection_counts[cls_name] = class_detection_counts.get(cls_name, 0) + 1

                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                line_counter.update(int(track_id), centroid)

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"ID {track_id} {cls_name} {conf:.2f}"
                cv2.putText(annotated, label, (int(x1), max(int(y1) - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        overlay = (f"FPS: {fps_meter.fps:.1f}   "
                   f"Crossed A->B: {line_counter.count_a_to_b}   B->A: {line_counter.count_b_to_a}")
        cv2.putText(annotated, overlay, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}  fps={fps_meter.fps:.1f}  crossings={line_counter.total}")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    summary = {
        "frames_processed": frame_idx,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        "detections_per_class": class_detection_counts,
        "line_crossings_a_to_b": line_counter.count_a_to_b,
        "line_crossings_b_to_a": line_counter.count_b_to_a,
    }
    with open(config.OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nAnnotated video saved to {args.output}")


if __name__ == "__main__":
    main()
