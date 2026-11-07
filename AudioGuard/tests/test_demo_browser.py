"""Optional browser regression test for five fake-microphone recording cycles.

Run against a locally running demo with AUDIOGUARD_DEMO_URL set. The default
pytest run skips this test unless the browser URL is explicitly configured.
"""
from __future__ import annotations

import math
import os
import struct
import wave

import pytest


@pytest.mark.skipif(
    not os.environ.get("AUDIOGUARD_DEMO_URL"),
    reason="Set AUDIOGUARD_DEMO_URL to a running local AudioGuard demo",
)
def test_fake_microphone_retains_preview_and_clears_after_analysis(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    wav_path = tmp_path / "fake_microphone.wav"
    sample_rate = 16000
    with wave.open(str(wav_path), "wb") as output:
        output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        frames = (
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(sample_rate * 3)
        )
        output.writeframes(b"".join(frames))

    errors = []
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav_path}",
            ],
        )
        try:
            context = browser.new_context(permissions=["microphone"])
            page = context.new_page()
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.on(
                "console",
                lambda message: errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.goto(os.environ["AUDIOGUARD_DEMO_URL"], wait_until="networkidle")

            for _ in range(5):
                record_source = page.locator('button[aria-label="Record audio"]')
                if record_source.count():
                    record_source.first.click()
                page.get_by_role("button", name="Record", exact=True).click()
                page.wait_for_timeout(2200)
                page.get_by_role("button", name="Stop", exact=True).first.click()
                page.wait_for_function(
                    "document.querySelector('#duration') && "
                    "document.querySelector('#duration').textContent.trim() !== '0:00'",
                    timeout=15000,
                )
                assert page.locator('audio[src^="blob:"]').count() > 0
                assert page.locator('button[aria-label="Play"]').count() > 0

                page.get_by_role("button", name="Analyse audio").click()
                page.wait_for_function(
                    "document.querySelectorAll('audio[src]').length === 0 && "
                    "document.querySelector('gradio-app')?.innerText.includes('predicted_label')",
                    timeout=60000,
                )
                assert "raw_logit" in page.locator("gradio-app").inner_text()
                assert page.locator("img").count() >= 2

            assert not errors
        finally:
            browser.close()
