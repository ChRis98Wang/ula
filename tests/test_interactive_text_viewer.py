import json


def test_interactive_session_reuses_viewer_and_generates_each_text(tmp_path):
    from upper_body_skeleton.interactive_text_viewer import run_interactive_session

    model = object()
    generated = []
    printed = []
    timer_values = iter([10.0, 10.25, 20.0, 20.5])

    class FakePlayer:
        def __init__(self):
            self.entered = 0
            self.exited = 0
            self.played = []

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited += 1
            return False

        def play_csv(self, csv_path, **kwargs):
            self.played.append((csv_path, kwargs))
            return {"frames_played": 3, "loops_completed": kwargs["loops"]}

    player = FakePlayer()

    def fake_generate(model_arg, **kwargs):
        assert model_arg is model
        assert kwargs["render"] is False
        assert kwargs["play"] is False
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "long_motion.csv"
        csv_path.write_text("time_sec\n0.0\n", encoding="utf-8")
        summary = {
            "output_dir": str(output_dir),
            "text": kwargs["text"],
            "csv": str(csv_path),
            "summary_json": str(output_dir / "summary.json"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        generated.append(kwargs)
        return summary

    result = run_interactive_session(
        model,
        player,
        ["开心地挥手", "", "紧张地解释", ":q", "不会执行"],
        output_root=tmp_path,
        session_name="session",
        generate_fn=fake_generate,
        writer=printed.append,
        fps=30.0,
        max_duration_sec=2.0,
        min_segment_sec=1.0,
        max_segment_sec=1.0,
        min_segments=2,
        max_segments=2,
        sampling_steps=2,
        device="cpu",
        seed=7,
        loops=1,
        realtime=False,
        timer=lambda: next(timer_values),
    )

    assert result["runs"] == 2
    assert player.entered == 1
    assert player.exited == 1
    assert [item["text"] for item in generated] == ["开心地挥手", "紧张地解释"]
    assert [call[1]["loops"] for call in player.played] == [1, 1]
    assert len(printed) == 2
    assert json.loads(printed[0])["generation_ms"] == 250.0
    assert json.loads(printed[1])["generation_ms"] == 500.0
    assert json.loads((tmp_path / "session" / "001" / "summary.json").read_text(encoding="utf-8"))[
        "generation_ms"
    ] == 250.0
    assert (tmp_path / "session" / "001" / "summary.json").is_file()
    assert (tmp_path / "session" / "002" / "summary.json").is_file()


def test_streaming_session_interrupts_current_motion_for_latest_text(tmp_path):
    from upper_body_skeleton.interactive_text_viewer import LatestTextInbox, run_streaming_session

    model = object()
    generated = []
    printed = []
    inbox = LatestTextInbox()
    inbox.submit("第一个动作")
    timer_values = iter([1.0, 1.1234, 2.0, 2.25])

    class FakePlayer:
        def __init__(self):
            self.entered = 0
            self.exited = 0
            self.played = []

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited += 1
            return False

        def play_csv(self, csv_path, **kwargs):
            self.played.append((csv_path, kwargs))
            if len(self.played) == 1:
                inbox.submit("第二个动作")
                kwargs["stop_event"].set()
                return {"frames_played": 1, "loops_completed": 0, "interrupted": True}
            return {"frames_played": 3, "loops_completed": 1, "interrupted": False}

    def fake_generate(model_arg, **kwargs):
        assert model_arg is model
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "long_motion.csv"
        csv_path.write_text("time_sec\n0.0\n", encoding="utf-8")
        summary = {"output_dir": str(output_dir), "text": kwargs["text"], "csv": str(csv_path)}
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        generated.append(kwargs["text"])
        return summary

    result = run_streaming_session(
        model,
        FakePlayer(),
        inbox,
        output_root=tmp_path,
        session_name="stream",
        generate_fn=fake_generate,
        writer=printed.append,
        fps=30.0,
        max_duration_sec=2.0,
        min_segment_sec=1.0,
        max_segment_sec=1.0,
        min_segments=1,
        max_segments=1,
        sampling_steps=2,
        device="cpu",
        seed=3,
        loops=1,
        realtime=False,
        idle_sleep_sec=0.0,
        timer=lambda: next(timer_values),
    )

    assert result["runs"] == 2
    assert generated == ["第一个动作", "第二个动作"]
    assert json.loads(printed[0])["interrupted"] is True
    assert json.loads(printed[0])["generation_ms"] == 123.4
    assert json.loads(printed[1])["interrupted"] is False
    assert json.loads(printed[1])["generation_ms"] == 250.0


def test_streaming_session_skips_stale_motion_when_text_changes_during_generation(tmp_path):
    from upper_body_skeleton.interactive_text_viewer import LatestTextInbox, run_streaming_session

    model = object()
    generated = []
    inbox = LatestTextInbox()
    inbox.submit("旧动作")

    class FakePlayer:
        def __init__(self):
            self.played = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def play_csv(self, csv_path, **kwargs):
            self.played.append(csv_path)
            return {"frames_played": 1, "loops_completed": 1, "interrupted": False}

    player = FakePlayer()

    def fake_generate(model_arg, **kwargs):
        assert model_arg is model
        if kwargs["text"] == "旧动作":
            inbox.submit("新动作")
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "long_motion.csv"
        csv_path.write_text("time_sec\n0.0\n", encoding="utf-8")
        generated.append((kwargs["text"], str(csv_path)))
        return {"output_dir": str(output_dir), "text": kwargs["text"], "csv": str(csv_path)}

    result = run_streaming_session(
        model,
        player,
        inbox,
        output_root=tmp_path,
        session_name="stream",
        generate_fn=fake_generate,
        writer=lambda _: None,
        idle_sleep_sec=0.0,
    )

    assert result["runs"] == 2
    assert [text for text, _ in generated] == ["旧动作", "新动作"]
    assert player.played == [generated[1][1]]


def test_streaming_session_exits_without_playback_when_closed_during_generation(tmp_path):
    from upper_body_skeleton.interactive_text_viewer import LatestTextInbox, run_streaming_session

    inbox = LatestTextInbox()
    inbox.submit("准备退出")

    class FakePlayer:
        def __init__(self):
            self.played = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def play_csv(self, csv_path, **kwargs):
            self.played.append(csv_path)
            return {"frames_played": 1, "loops_completed": 1, "interrupted": False}

    player = FakePlayer()

    def fake_generate(model_arg, **kwargs):
        inbox.close()
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "long_motion.csv"
        csv_path.write_text("time_sec\n0.0\n", encoding="utf-8")
        return {"output_dir": str(output_dir), "text": kwargs["text"], "csv": str(csv_path)}

    result = run_streaming_session(
        object(),
        player,
        inbox,
        output_root=tmp_path,
        session_name="stream",
        generate_fn=fake_generate,
        writer=lambda _: None,
        idle_sleep_sec=0.0,
    )

    assert result["runs"] == 1
    assert player.played == []
