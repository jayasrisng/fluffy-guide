# fluffy-guide Case Study

## Summary

fluffy-guide is a real-time lecture copilot. It maintains rolling lecture context and lets a learner ask quiet questions while the lecture continues.

The project explores live comprehension support rather than after-the-fact summarization.

## Problem

In a lecture, confusion is time-sensitive. If a learner misses a definition or gets lost during a transition, waiting until the end can make the rest of the lecture harder to follow.

The project asks:

> Can an AI assistant help a learner recover understanding in the moment without interrupting the speaker?

## Approach

The prototype uses two agents:

1. A listener agent captures/transcribes lecture audio and maintains recent context.
2. A guide agent answers learner questions using the rolling transcript and summary.

This keeps the assistant grounded in what was just said.

## Technical stack

- Python
- Streamlit
- faster-whisper
- OpenAI API
- Environment-based configuration

## Design decisions

### Separate speaker and learner

The lecturer is not treated as the user. This allows the learner to interact silently without changing the room dynamic.

### Maintain recent context

A rolling buffer is more useful than one-off transcription because questions often depend on what happened over the last few minutes.

### Support demo mode

Live audio can be unreliable in noisy spaces. Demo mode makes the interaction easier to test and present.

## Challenges

### Audio quality

Microphone quality, speaker distance, and background noise can all affect transcription accuracy.

### Grounding

Answers should stay tied to recent lecture context. Otherwise the assistant becomes a generic chatbot instead of a learning copilot.

### Consent and privacy

Lecture audio and transcripts can be sensitive. The app needs clear consent and data-handling boundaries before real deployment.

## What this demonstrates

- Real-time AI interaction design.
- Multi-agent context management.
- Practical speech-to-text pipeline integration.
- Learning-support product thinking.

## Future work

- Add transcript snippets as citations for answers.
- Add slide/image context.
- Add opt-in transcript export.
- Add privacy and consent controls.
- Explore XR gaze or spatial input for silent questions.
