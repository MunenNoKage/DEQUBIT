# Recording a demo for README.md

The root README has a spot for a demo GIF or screenshot near the top. A short recording
is worth more than a paragraph of description: a reader can tell what the project does
in about five seconds instead of reading the whole README first.

## Quick option: a static screenshot

Run the app, get it into a state that shows a real result rather than an empty form,
and take a screenshot:

```bash
streamlit run app/streamlit_app.py
```

Save it as `results/demo/demo.png` and reference it from the README, with alt text
that says what is happening in the image:

```markdown
![<what the screenshot shows>](results/demo/demo.png)
```

Good enough for most submissions, and far less effort than a recording.

## A recorded GIF

Any screen recorder that exports GIF will do — ScreenToGif on Windows, Kap or
LICEcap on macOS, Peek on Linux. Keep it under about ten seconds and show one
complete action from start to result.

Keep the file small. A GIF over a few megabytes makes the repo unpleasant to clone and
is the one committed binary most likely to grow without anyone noticing — crop to the
app window, drop the frame rate, and trim dead time at the start.

## Write a real alt text

Describe what is happening in the image, not just "Demo". Someone reading with a
screen reader, or looking at the README when the image fails to load, then gets the
same information — and it costs one sentence.
