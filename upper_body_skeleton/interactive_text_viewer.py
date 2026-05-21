#!/usr/bin/env python3
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from upper_body_skeleton.long_emotion_infer import REPO_ROOT, generate_long_emotion_motion
from upper_body_skeleton.mujoco_playback import MujocoMotionPlayer
from upper_body_skeleton.ula_infer import load_model
from upper_body_skeleton.ula_training import choose_device


DEFAULT_CHECKPOINT = REPO_ROOT / "training" / "runs" / "ula_fm_0514_close_v2_1m" / "ula_fm_checkpoint.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "deliverables" / "interactive_text_tests"


def _default_session_name():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class LatestTextInbox:
    def __init__(self):
        self._condition = threading.Condition()
        self._latest_text = None
        self._closed = False
        self._stop_event = None

    def submit(self, text):
        text = str(text).strip()
        if not text:
            return
        with self._condition:
            self._latest_text = text
            if self._stop_event is not None:
                self._stop_event.set()
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            if self._stop_event is not None:
                self._stop_event.set()
            self._condition.notify_all()

    def set_stop_event(self, stop_event):
        with self._condition:
            self._stop_event = stop_event
            if self._latest_text is not None or self._closed:
                self._stop_event.set()

    def clear_stop_event(self, stop_event):
        with self._condition:
            if self._stop_event is stop_event:
                self._stop_event = None

    def pop_latest(self, timeout=None):
        with self._condition:
            if self._latest_text is None and not self._closed:
                self._condition.wait(timeout=timeout)
            if self._latest_text is None:
                return None
            text = self._latest_text
            self._latest_text = None
            return text

    @property
    def closed(self):
        with self._condition:
            return self._closed


def _is_quit_text(text):
    return text in {":q", ":quit", "quit", "exit"}


def _write_summary(summary, viewer, run_count, text, writer):
    summary["viewer"] = viewer
    summary_path = Path(summary["output_dir"]) / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    writer(
        json.dumps(
            {
                "index": run_count,
                "text": text,
                "summary_json": str(summary_path),
                "generation_ms": summary.get("generation_ms"),
                "interrupted": bool(viewer.get("interrupted", False)),
            },
            ensure_ascii=False,
        )
    )


def _generate_motion(
    model,
    generate_fn,
    text,
    output_dir,
    run_count,
    *,
    fps,
    max_duration_sec,
    min_segment_sec,
    max_segment_sec,
    min_segments,
    max_segments,
    sampling_steps,
    device,
    seed,
    max_velocity_rad_s,
    smooth_window,
    timer=time.perf_counter,
):
    started = timer()
    summary = generate_fn(
        model,
        text=text,
        output_dir=output_dir,
        fps=fps,
        max_duration_sec=max_duration_sec,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        min_segments=min_segments,
        max_segments=max_segments,
        sampling_steps=sampling_steps,
        device=device,
        seed=seed + run_count - 1 if seed is not None else None,
        render=False,
        max_velocity_rad_s=max_velocity_rad_s,
        smooth_window=smooth_window,
        play=False,
    )
    finished = timer()
    summary["generation_ms"] = round((finished - started) * 1000.0, 3)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return summary


def run_interactive_session(
    model,
    player,
    input_lines,
    *,
    output_root=DEFAULT_OUTPUT_ROOT,
    session_name=None,
    generate_fn=generate_long_emotion_motion,
    writer=print,
    fps=30.0,
    max_duration_sec=12.0,
    min_segment_sec=3.0,
    max_segment_sec=3.0,
    min_segments=4,
    max_segments=4,
    sampling_steps=16,
    device="cpu",
    seed=7,
    max_velocity_rad_s=3.0,
    smooth_window=5,
    loops=1,
    realtime=True,
    timer=time.perf_counter,
):
    session_dir = Path(output_root) / (session_name or _default_session_name())
    run_count = 0
    with player as active_player:
        for raw_line in input_lines:
            text = raw_line.strip()
            if not text:
                continue
            if _is_quit_text(text):
                break
            run_count += 1
            output_dir = session_dir / f"{run_count:03d}"
            summary = _generate_motion(
                model,
                generate_fn,
                text,
                output_dir,
                run_count,
                fps=fps,
                max_duration_sec=max_duration_sec,
                min_segment_sec=min_segment_sec,
                max_segment_sec=max_segment_sec,
                min_segments=min_segments,
                max_segments=max_segments,
                sampling_steps=sampling_steps,
                device=device,
                seed=seed,
                max_velocity_rad_s=max_velocity_rad_s,
                smooth_window=smooth_window,
                timer=timer,
            )
            viewer = active_player.play_csv(summary["csv"], loops=loops, realtime=realtime)
            _write_summary(summary, viewer, run_count, text, writer)
    return {"session_dir": str(session_dir), "runs": run_count}


def run_streaming_session(
    model,
    player,
    inbox,
    *,
    output_root=DEFAULT_OUTPUT_ROOT,
    session_name=None,
    generate_fn=generate_long_emotion_motion,
    writer=print,
    fps=30.0,
    max_duration_sec=12.0,
    min_segment_sec=3.0,
    max_segment_sec=3.0,
    min_segments=4,
    max_segments=4,
    sampling_steps=16,
    device="cpu",
    seed=7,
    max_velocity_rad_s=3.0,
    smooth_window=5,
    loops=1,
    realtime=True,
    idle_sleep_sec=None,
    timer=time.perf_counter,
):
    session_dir = Path(output_root) / (session_name or _default_session_name())
    run_count = 0
    with player as active_player:
        while True:
            text = inbox.pop_latest(timeout=idle_sleep_sec)
            if text is None:
                break
            if _is_quit_text(text):
                inbox.close()
                break

            run_count += 1
            output_dir = session_dir / f"{run_count:03d}"
            summary = _generate_motion(
                model,
                generate_fn,
                text,
                output_dir,
                run_count,
                fps=fps,
                max_duration_sec=max_duration_sec,
                min_segment_sec=min_segment_sec,
                max_segment_sec=max_segment_sec,
                min_segments=min_segments,
                max_segments=max_segments,
                sampling_steps=sampling_steps,
                device=device,
                seed=seed,
                max_velocity_rad_s=max_velocity_rad_s,
                smooth_window=smooth_window,
                timer=timer,
            )

            pending_text = inbox.pop_latest(timeout=0.0)
            if pending_text is not None:
                if _is_quit_text(pending_text):
                    inbox.close()
                    break
                inbox.submit(pending_text)
                continue
            if inbox.closed:
                break

            stop_event = threading.Event()
            inbox.set_stop_event(stop_event)
            try:
                viewer = active_player.play_csv(
                    summary["csv"],
                    loops=loops,
                    realtime=realtime,
                    stop_event=stop_event,
                )
            finally:
                inbox.clear_stop_event(stop_event)
            _write_summary(summary, viewer, run_count, text, writer)
            if inbox.closed:
                break
    return {"session_dir": str(session_dir), "runs": run_count}


def _read_input_loop(input_lines, inbox):
    for raw_line in input_lines:
        text = raw_line.strip()
        if not text:
            continue
        if _is_quit_text(text):
            inbox.close()
            return
        inbox.submit(text)
    inbox.close()


def _stdin_lines(prompt):
    while True:
        try:
            yield input(prompt)
        except EOFError:
            return


def main():
    parser = argparse.ArgumentParser(description="Stream text prompts into one reusable MuJoCo viewer")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--session-name")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration-sec", type=float, default=12.0)
    parser.add_argument("--min-segment-sec", type=float, default=3.0)
    parser.add_argument("--max-segment-sec", type=float, default=3.0)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--loops", type=int, default=1, help="Playback loops per entered text; 1 returns to prompt")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--simplified", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    player = MujocoMotionPlayer(fps=args.fps, simplified=args.simplified)
    inbox = LatestTextInbox()
    input_thread = threading.Thread(target=_read_input_loop, args=(_stdin_lines("> "), inbox), daemon=True)
    input_thread.start()
    print("输入 text 后回车播放；播放中输入新 text 会打断当前动作并切换；输入 :q 退出。")
    result = run_streaming_session(
        model,
        player,
        inbox,
        output_root=args.output_root,
        session_name=args.session_name,
        fps=args.fps,
        max_duration_sec=args.max_duration_sec,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        sampling_steps=args.sampling_steps,
        device=device,
        seed=args.seed,
        max_velocity_rad_s=args.max_velocity_rad_s,
        smooth_window=args.smooth_window,
        loops=args.loops,
        realtime=not args.no_realtime,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
